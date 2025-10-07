# frozen_string_literal: true

require 'etc'

module Rllama
  class Context
    attr_reader :messages, :n_ctx, :n_batch, :n_past

    def initialize(model, embeddings: false, n_ctx: nil, n_batch: nil, n_threads: Etc.nprocessors)
      @model = model
      @n_ctx = n_ctx
      @n_batch = n_batch
      @embeddings = embeddings

      @ctx_params = Cpp.llama_context_default_params

      @ctx_params[:n_ctx] = @n_ctx if @n_ctx
      @ctx_params[:n_batch] = @n_batch if @n_batch

      @ctx_params[:n_threads] = n_threads
      @ctx_params[:n_threads_batch] = n_threads

      if @embeddings
        seq_cap = @model.n_seq_max

        if @n_batch&.positive? && seq_cap&.positive?
          @ctx_params[:n_seq_max] = [@n_batch, seq_cap].min
        elsif seq_cap&.positive?
          @ctx_params[:n_seq_max] = seq_cap
        end

        @ctx_params[:embeddings] = true
        @ctx_params[:kv_unified] = true
        @ctx_params[:n_ubatch] = @n_batch if @n_batch&.positive?
      end

      @pointer = Cpp.llama_init_from_model(model.pointer, @ctx_params)

      raise Error, 'Failed to create the llama_context' if @pointer.null?

      @n_past = 0
      @messages = []
    end

    def generate(message, role: 'user', max_tokens: @n_ctx, temperature: 0.8, top_k: 40, top_p: 0.95, min_p: 0.05,
                 seed: nil, system: nil)
      @messages << { role: 'system', content: system } if system && @messages.empty?

      if message.is_a?(Array)
        @messages.push(*message)
      elsif message.is_a?(Hash)
        @messages.push(message)
      else
        @messages << { role: role, content: message }
      end

      prompt_string = @model.build_chat_template(@messages)

      n_prompt_tokens = -Cpp.llama_tokenize(@model.vocab, prompt_string, prompt_string.bytesize, nil, 0, true, true)

      raise Error, 'Prompt is too long.' if n_prompt_tokens.negative?

      prompt_tokens_ptr = FFI::MemoryPointer.new(:int32, n_prompt_tokens)
      tokens_written = Cpp.llama_tokenize(@model.vocab, prompt_string, prompt_string.bytesize, prompt_tokens_ptr,
                                          n_prompt_tokens, true, true)

      raise Error, 'Failed to tokenize prompt.' if tokens_written.negative?

      new_token_count = tokens_written - @n_past

      if new_token_count.positive?
        new_tokens_ptr = prompt_tokens_ptr + (@n_past * FFI.type_size(:int32))

        batch = Cpp.llama_batch_get_one(new_tokens_ptr, new_token_count)

        raise Error, 'llama_decode failed.' if Cpp.llama_decode(@pointer, batch) != 0

        @n_past = tokens_written
      end

      chain_params = Cpp.llama_sampler_chain_default_params
      sampler_chain = Cpp.llama_sampler_chain_init(chain_params)

      Cpp.llama_sampler_chain_add(sampler_chain, Cpp.llama_sampler_init_min_p(min_p, 1)) if min_p
      Cpp.llama_sampler_chain_add(sampler_chain, Cpp.llama_sampler_init_top_k(top_k)) if top_k&.positive?
      Cpp.llama_sampler_chain_add(sampler_chain, Cpp.llama_sampler_init_top_p(top_p, 1)) if top_p && top_p < 1.0
      if temperature&.positive?
        Cpp.llama_sampler_chain_add(sampler_chain,
                                    Cpp.llama_sampler_init_temp(temperature))
      end

      is_probabilistic = temperature&.positive? || top_k&.positive? || (top_p && top_p < 1.0) || !min_p.nil?
      rng_seed = seed || (Random.new_seed & 0xFFFFFFFF)

      if is_probabilistic
        Cpp.llama_sampler_chain_add(sampler_chain, Cpp.llama_sampler_init_dist(rng_seed))
      else
        Cpp.llama_sampler_chain_add(sampler_chain, Cpp.llama_sampler_init_greedy)
      end

      n_decoded = 0

      generated_text = ''.b

      assistant_message = { role: 'assistant', content: generated_text }

      @messages << assistant_message

      start_time = Time.now

      loop do
        break if n_decoded >= max_tokens

        new_token_id = Cpp.llama_sampler_sample(sampler_chain, @pointer, -1)

        break if Cpp.llama_vocab_is_eog(@model.vocab, new_token_id)

        buffer = FFI::MemoryPointer.new(:char, 256)
        n_chars = Cpp.llama_token_to_piece(@model.vocab, new_token_id, buffer, buffer.size, 0, true)

        if n_chars >= 0
          piece_bytes = buffer.read_string(n_chars)
          utf8_piece = piece_bytes.force_encoding(Encoding::UTF_8)
          generated_text << utf8_piece
          yield utf8_piece if block_given?
        end

        token_ptr = FFI::MemoryPointer.new(:int32, 1).put_int32(0, new_token_id)
        batch = Cpp.llama_batch_get_one(token_ptr, 1)

        raise Error, 'context length has been exceeded' if @n_past >= @n_ctx
        raise Error, 'llama_decode failed.' if Cpp.llama_decode(@pointer, batch) != 0

        @n_past += 1
        n_decoded += 1
      end

      end_time = Time.now

      duration = end_time - start_time

      tps = n_decoded.positive? && duration.positive? ? n_decoded / duration : 0

      Cpp.llama_sampler_free(sampler_chain)

      Result.new(
        text: generated_text,
        stats: {
          duration:,
          tokens_generated: n_decoded,
          tps:,
          seed: rng_seed
        }
      )
    end
    alias message generate

    def embed(strings_or_tokens, normalize: true, batch_size: 512)
      is_tokens = strings_or_tokens.is_a?(Array) &&
                  (strings_or_tokens[0].is_a?(Integer) ||
                   (strings_or_tokens[0].is_a?(Array) && strings_or_tokens[0][0].is_a?(Integer)))

      input_is_array = is_tokens ? strings_or_tokens[0].is_a?(Array) : strings_or_tokens.is_a?(Array)

      normalized_inputs = input_is_array ? strings_or_tokens : [strings_or_tokens]

      tokenized_strings =
        if is_tokens
          input_is_array ? strings_or_tokens : [strings_or_tokens]
        else
          normalized_inputs.map { |text| @model.tokenize(text) }
        end

      max_tokens_in_prompt = tokenized_strings.map(&:length).max || 0

      if max_tokens_in_prompt > batch_size
        raise Error, "batch_size (#{batch_size}) is smaller than the longest prompt (#{max_tokens_in_prompt} tokens)."
      end

      if max_tokens_in_prompt > @n_batch
        raise Error, "Context n_batch (#{@n_batch}) is smaller than the longest " \
                     "prompt (#{max_tokens_in_prompt} tokens). Increase batch_size when calling embed."
      end

      all_embeddings = []
      batch = Cpp.llama_batch_init(batch_size, 0, 1)
      prompts_in_batch = []
      current_batch_token_count = 0

      process_batch = lambda do
        next if prompts_in_batch.empty?

        batch[:n_tokens] = current_batch_token_count

        memory_ptr = Cpp.llama_get_memory(@pointer)
        Cpp.llama_memory_clear(memory_ptr, true) unless memory_ptr.null?

        raise Error, 'llama_decode failed' unless Cpp.llama_decode(@pointer, batch).zero?

        prompts_in_batch.each do |seq_id_in_batch|
          embd_ptr = Cpp.llama_get_embeddings_seq(@pointer, seq_id_in_batch)

          raise Error, 'Failed to get embedding' if embd_ptr.null?

          embedding = embd_ptr.read_array_of_float(@model.n_embd)

          all_embeddings << (normalize ? normalize_embedding(embedding) : embedding)
        end

        current_batch_token_count = 0

        prompts_in_batch.clear
      end

      tokenized_strings.each do |tokens|
        batch_full = (current_batch_token_count + tokens.size) > batch_size
        seq_limit_reached = prompts_in_batch.size >= @model.n_seq_max
        process_batch.call if !prompts_in_batch.empty? && (batch_full || seq_limit_reached)

        seq_id = prompts_in_batch.size
        prompts_in_batch << seq_id

        tokens.each_with_index do |token_id, pos|
          idx = current_batch_token_count

          batch[:token].put_int32(idx * FFI.type_size(:int32), token_id)
          batch[:pos].put_int32(idx * FFI.type_size(:int32), pos)
          batch[:n_seq_id].put_int32(idx * FFI.type_size(:int32), 1)
          batch[:seq_id].get_pointer(idx * FFI::Pointer.size).put_int32(0, seq_id)
          batch[:logits].put_int8(idx, pos == tokens.size - 1 ? 1 : 0)

          current_batch_token_count += 1
        end
      end

      process_batch.call

      Cpp.llama_batch_free(batch)

      input_is_array ? all_embeddings : all_embeddings[0]
    end

    def embeddings?
      @embeddings
    end

    def close
      Cpp.llama_free(@pointer)
    end

    def norm(vec)
      Math.sqrt(vec.sum { |x| x**2 })
    end

    def normalize_embedding(vec)
      n = norm(vec)

      return vec if n.zero?

      vec.map { |x| x / n }
    end
  end
end
