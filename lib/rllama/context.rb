# frozen_string_literal: true

require 'etc'
require 'json'

module Rllama
  class Context
    TOOL_CALL_ID_PREFIX = 'call_'
    GRAMMAR_TYPE_TOOL_CALLS = 3

    attr_reader :messages, :n_ctx, :n_batch, :n_past

    def initialize(model, embeddings: false, n_ctx: nil, n_batch: nil, n_threads: Etc.nprocessors,
                   system: nil, tools: nil, reasoning: false)
      @model = model
      @tools = tools
      @reasoning = reasoning
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

      @n_ctx = Cpp.llama_n_ctx(@pointer)
      @n_batch = Cpp.llama_n_batch(@pointer)

      @n_past = 0
      @cache_tokens = []
      @tool_call_count = 0
      @messages = []
      @messages << { role: 'system', content: system } if system

      prefill if !@embeddings && (system || tools)
    end

    def generate(message, role: 'user', max_tokens: @n_ctx, temperature: nil, top_k: nil, top_p: nil, min_p: nil,
                 seed: nil, system: nil, tools: nil, reasoning: nil, &block)
      temperature, top_k, top_p, min_p = resolve_sampling(temperature, top_k, top_p, min_p)

      @tools = tools unless tools.nil?
      @reasoning = reasoning unless reasoning.nil?

      if system
        if @messages.dig(0, :role).to_s == 'system'
          @messages.first[:content] = system
        else
          @messages.unshift(role: 'system', content: system)
        end
      end

      if message.is_a?(Array)
        @messages.push(*message)
      elsif message.is_a?(Hash)
        @messages.push(message)
      else
        @messages << { role: role, content: message }
      end

      applied = @model.apply_chat_template(@messages, tools: @tools, enable_thinking: @reasoning)

      decode_prompt(applied['prompt'])

      rng_seed = seed || (Random.new_seed & 0xFFFFFFFF)

      sampler = Common.sampler_init(@model.pointer, sampler_params(temperature, top_k, top_p, min_p, rng_seed, applied))

      stops = applied['additional_stops'].to_a.map(&:b)
      max_stop = stops.map(&:bytesize).max || 0

      n_decoded = 0
      n_yielded = 0

      generated_text = ''.b

      assistant_message = { role: 'assistant', content: generated_text }

      @messages << assistant_message

      start_time = Time.now

      loop do
        break if n_decoded >= max_tokens

        new_token_id = Common.sampler_sample(sampler, @pointer)

        Common.sampler_accept(sampler, new_token_id)

        break if Cpp.llama_vocab_is_eog(@model.vocab, new_token_id)

        buffer = FFI::MemoryPointer.new(:char, 256)
        n_chars = Cpp.llama_token_to_piece(@model.vocab, new_token_id, buffer, buffer.size, 0, true)

        if n_chars >= 0
          stop_at, n_yielded = emit_piece(generated_text, buffer.read_string(n_chars), stops, max_stop, n_yielded,
                                          &block)

          break if stop_at
        end

        token_ptr = FFI::MemoryPointer.new(:int32, 1).put_int32(0, new_token_id)
        batch = Cpp.llama_batch_get_one(token_ptr, 1)

        raise Error, 'context length has been exceeded' if @n_past >= @n_ctx
        raise Error, 'llama_decode failed.' if Cpp.llama_decode(@pointer, batch) != 0

        @n_past += 1
        @cache_tokens << new_token_id
        n_decoded += 1
      end

      if block_given? && n_yielded < generated_text.bytesize
        yield generated_text.byteslice(n_yielded..).force_encoding(Encoding::UTF_8)
      end

      generated_text.force_encoding(Encoding::UTF_8)

      end_time = Time.now

      duration = end_time - start_time

      tps = n_decoded.positive? && duration.positive? ? n_decoded / duration : 0

      Common.sampler_free(sampler)

      text = generated_text
      reasoning = nil
      tool_calls = []

      if @tools || @reasoning
        parsed = Common.chat_parse(generated_text, applied, reasoning: @reasoning)

        reasoning = parsed['reasoning_content']
        reasoning = nil if reasoning&.empty?
        tool_calls = extract_tool_calls(parsed, assistant_message) if @tools

        if @reasoning && tool_calls.empty?
          text = parsed['content'].to_s
          assistant_message[:content] = text
        end
      end

      Result.new(
        text:,
        reasoning:,
        tool_calls:,
        stats: {
          duration:,
          tokens_generated: n_decoded,
          tps:,
          seed: rng_seed,
          temperature:,
          top_k:,
          top_p:,
          min_p:
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

    def prefill
      decode_prompt(@model.build_chat_template(@messages, tools: @tools, add_generation_prompt: false,
                                                          enable_thinking: @reasoning))
    rescue StandardError
      nil
    end

    def extract_tool_calls(parsed, assistant_message)
      calls = (parsed['tool_calls'] || []).map do |call|
        id = call['id'].to_s
        id = "#{TOOL_CALL_ID_PREFIX}#{@tool_call_count += 1}" if id.empty?

        arguments = begin
          JSON.parse(call.dig('function', 'arguments').to_s)
        rescue JSON::ParserError
          {}
        end

        { name: call.dig('function', 'name'), arguments:, id: }
      end

      return [] if calls.empty?

      assistant_message.replace(
        role: 'assistant',
        content: nil,
        tool_calls: calls.map do |call|
          { type: 'function', id: call[:id],
            function: { name: call[:name], arguments: call[:arguments].to_json } }
        end
      )

      calls
    end

    def decode_prompt(prompt_string)
      n_prompt_tokens = -Cpp.llama_tokenize(@model.vocab, prompt_string, prompt_string.bytesize, nil, 0, true, true)

      raise Error, 'Prompt is too long.' if n_prompt_tokens.negative?

      prompt_tokens_ptr = FFI::MemoryPointer.new(:int32, n_prompt_tokens)
      tokens_written = Cpp.llama_tokenize(@model.vocab, prompt_string, prompt_string.bytesize, prompt_tokens_ptr,
                                          n_prompt_tokens, true, true)

      raise Error, 'Failed to tokenize prompt.' if tokens_written.negative?

      prompt_tokens = prompt_tokens_ptr.read_array_of_int32(tokens_written)

      common = common_prefix_length(prompt_tokens)

      if common < @cache_tokens.length
        memory = Cpp.llama_get_memory(@pointer)

        if memory.null? || !Cpp.llama_memory_seq_rm(memory, 0, common, -1)
          Cpp.llama_memory_clear(memory, true) unless memory.null?

          common = 0
        end

        @n_past = common
      end

      while tokens_written > @n_past
        n_eval = [tokens_written - @n_past, @n_batch].min

        new_tokens_ptr = prompt_tokens_ptr + (@n_past * FFI.type_size(:int32))

        batch = Cpp.llama_batch_get_one(new_tokens_ptr, n_eval)

        raise Error, 'llama_decode failed.' if Cpp.llama_decode(@pointer, batch) != 0

        @n_past += n_eval
      end

      @cache_tokens = prompt_tokens
    end

    def common_prefix_length(tokens)
      limit = [@cache_tokens.length, tokens.length - 1].min

      common = 0
      common += 1 while common < limit && @cache_tokens[common] == tokens[common]

      common
    end

    def sampler_params(temperature, top_k, top_p, min_p, seed, applied)
      params = {
        temp: temperature || 0.0,
        top_k: top_k&.positive? ? top_k : 0,
        top_p: top_p || 1.0,
        min_p: min_p || 0.0,
        seed:
      }

      grammar = applied['grammar'].to_s

      unless grammar.empty?
        params.merge!(grammar:, grammar_type: GRAMMAR_TYPE_TOOL_CALLS,
                      grammar_lazy: applied['grammar_lazy'],
                      grammar_triggers: applied['grammar_triggers'],
                      generation_prompt: applied['generation_prompt'])
      end

      params
    end

    def emit_piece(text, piece, stops, max_stop, n_yielded, &block)
      tail_from = [text.bytesize - max_stop + 1, 0].max
      text << piece

      stop_at = max_stop.positive? ? stops.filter_map { |stop| text.index(stop, tail_from) }.min : nil
      text.slice!(stop_at..) if stop_at

      if block
        safe = text.bytesize - (stop_at ? 0 : stop_holdback(text, stops))

        if safe > n_yielded
          yield(text.byteslice(n_yielded, safe - n_yielded).force_encoding(Encoding::UTF_8))
          n_yielded = safe
        end
      end

      [stop_at, n_yielded]
    end

    def stop_holdback(text, stops)
      stops.map do |stop|
        (stop.bytesize - 1).downto(1).find { |n| text.end_with?(stop.byteslice(0, n)) } || 0
      end.max || 0
    end

    def resolve_sampling(temperature, top_k, top_p, min_p)
      defaults = @model.sampling_defaults

      [
        temperature.nil? ? defaults[:temperature] : temperature,
        top_k.nil? ? defaults[:top_k] : top_k,
        top_p.nil? ? defaults[:top_p] : top_p,
        min_p.nil? ? defaults[:min_p] : min_p
      ]
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
