# frozen_string_literal: true

module Rllama
  class Model
    DEFAULT_CONTEXT_LENGTH = 2**13

    DEFAULT_SAMPLING = { temperature: 0.8, top_k: 40, top_p: 0.95, min_p: 0.05 }.freeze

    attr_reader :pointer

    def initialize(path_or_name, dir: nil)
      resolved_path = Loader.resolve(path_or_name, dir:)

      model_params = Cpp.llama_model_default_params

      @pointer = Cpp.llama_model_load_from_file(resolved_path, model_params)

      raise Error, "Unable to load model from #{resolved_path}" if @pointer.null?
    end

    def chat_template
      @chat_template ||= Cpp.llama_model_chat_template(@pointer, nil)
    end

    def vocab
      @vocab ||= Cpp.llama_model_get_vocab(@pointer)
    end

    def n_embd
      @n_embd ||= Cpp.llama_model_n_embd(@pointer)
    end

    def n_seq_max
      @n_seq_max ||= Cpp.llama_max_parallel_sequences
    end

    def n_ctx_train
      @n_ctx_train ||= Cpp.llama_model_n_ctx_train(@pointer)
    end

    def meta(key)
      buffer = FFI::MemoryPointer.new(:char, 256)
      length = Cpp.llama_model_meta_val_str(@pointer, key.to_s, buffer, buffer.size)

      length.negative? ? nil : buffer.read_string
    end

    def sampling_defaults
      @sampling_defaults ||= {
        temperature: fetch_meta_float('general.sampling.temp') || DEFAULT_SAMPLING[:temperature],
        top_k: fetch_meta_int('general.sampling.top_k') || DEFAULT_SAMPLING[:top_k],
        top_p: fetch_meta_float('general.sampling.top_p') || DEFAULT_SAMPLING[:top_p],
        min_p: fetch_meta_float('general.sampling.min_p') || DEFAULT_SAMPLING[:min_p]
      }
    end

    def bos_token
      @bos_token ||= token_to_string(Cpp.llama_vocab_bos(vocab))
    end

    def eos_token
      @eos_token ||= token_to_string(Cpp.llama_vocab_eos(vocab))
    end

    def generate(prompt, max_tokens: DEFAULT_CONTEXT_LENGTH, temperature: nil, top_k: nil, top_p: nil, min_p: nil,
                 seed: nil, system: nil, tools: nil, reasoning: nil, &block)
      init_context(n_ctx: max_tokens) do |ctx|
        ctx.generate(prompt, max_tokens: ctx.n_ctx,
                             temperature:, top_k:, top_p:, seed:, system:, min_p:, tools:, reasoning:,
                     &block)
      end
    end
    alias message generate

    def embed(prompt, normalize: true, batch_size: 512, &block)
      inputs = prompt.is_a?(Array) ? prompt : [prompt]

      tokenized_inputs = inputs.map { |text| tokenize(text, max_tokens: n_ctx_train) }
      max_token_length = tokenized_inputs.map(&:length).max || 0

      effective_batch_size = [batch_size, max_token_length].max
      effective_ctx = [n_ctx_train, max_token_length].min

      init_embedding_context(n_ctx: effective_ctx, n_batch: effective_batch_size) do |ctx|
        inputs = prompt.is_a?(Array) ? tokenized_inputs : tokenized_inputs[0]

        ctx.embed(inputs, normalize:, batch_size: effective_batch_size, &block)
      end
    end

    def tokenize(text, max_tokens: nil)
      size = text.bytesize + 2

      tokens_ptr = FFI::MemoryPointer.new(:int32, size)
      count = Cpp.llama_tokenize(vocab, text, text.bytesize, tokens_ptr, size, true, false)

      raise Error, "Failed to tokenize text: '#{text}'" if count.negative?

      tokens_ptr.read_array_of_int32([count, max_tokens].compact.min)
    end

    def close
      if @chat_templates
        Common.chat_templates_free(@chat_templates)

        @chat_templates = nil
      end

      Cpp.llama_model_free(@pointer)
    end

    def init_context(embeddings: false, n_ctx: DEFAULT_CONTEXT_LENGTH, n_batch: 512, system: nil, tools: nil,
                     reasoning: false)
      context = Context.new(self, embeddings:, n_ctx:, n_batch:, system:, tools:, reasoning:)

      if block_given?
        result = yield context

        context.close

        return result
      end

      context
    end

    def init_embedding_context(n_ctx: n_ctx_train, n_batch: 512, &)
      init_context(embeddings: true, n_ctx:, n_batch:, &)
    end

    def apply_chat_template(messages, tools: nil, add_generation_prompt: true, enable_thinking: false)
      raise Error, 'Model does not provide a chat template' if chat_template.nil? || chat_template.empty?

      Common.chat_apply(chat_templates, messages:, tools:, add_generation_prompt:, enable_thinking:,
                                        add_bos: Cpp.llama_vocab_get_add_bos(vocab),
                                        add_eos: Cpp.llama_vocab_get_add_eos(vocab))
    end

    def build_chat_template(messages, tools: nil, add_generation_prompt: true, enable_thinking: false)
      apply_chat_template(messages, tools:, add_generation_prompt:, enable_thinking:)['prompt']
    end

    private

    def fetch_meta_float(key)
      value = meta(key)

      value && Float(value, exception: false)
    end

    def fetch_meta_int(key)
      value = meta(key)

      return nil unless value

      Integer(value, exception: false) || Float(value, exception: false)&.to_i
    end

    def chat_templates
      @chat_templates ||= Common.chat_templates_init(@pointer, chat_template, bos_token, eos_token)
    end

    def token_to_string(token_id)
      buf = FFI::MemoryPointer.new(:char, 256)
      n = Cpp.llama_token_to_piece(vocab, token_id, buf, buf.size, 0, true)

      n.positive? ? buf.read_string(n) : ''
    end
  end
end
