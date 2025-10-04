module Rllama
  class Model
    DEFAULT_CONTEXT_LENGTH = 2 ** 13

    attr_reader :pointer

    def initialize(path)
      model_params = Cpp.llama_model_default_params

      @pointer = Cpp.llama_model_load_from_file(path, model_params)

      raise Error, "Unable to load model from #{path}" if @pointer.null?
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
      @llama_max_parallel_sequences ||= Cpp.llama_max_parallel_sequences
    end

    def n_ctx_train
      @n_ctx_train ||= Cpp.llama_model_n_ctx_train(@pointer)
    end

    def generate(prompt, max_tokens: DEFAULT_CONTEXT_LENGTH, temperature: 0.8, top_k: 40, top_p: 0.95, min_p: 0.05, seed: nil, system: nil, &block)
      init_context(n_ctx: max_tokens) do |ctx|
        ctx.generate(prompt, max_tokens: ctx.n_ctx,
                             temperature:, top_k:, top_p:, seed:, system:, min_p:,
                             &block)
      end
    end
    alias message generate

    def embed(prompt, normalize: true, batch_size: 512, &block)
      init_embedding_context do |ctx|
        ctx.embed(prompt, normalize:, batch_size:, &block)
      end
    end

    def close
      Cpp.llama_model_free(@pointer)
    end

    def init_context(embeddings: false, n_ctx: DEFAULT_CONTEXT_LENGTH, n_batch: 512)
      context = Context.new(self, embeddings:, n_ctx:, n_batch:)

      if block_given?
        result = yield context

        context.close

        return result
      end

      context
    end

    def init_embedding_context(n_ctx: 2048, n_batch: 512, &block)
      init_context(embeddings: true, n_ctx:, n_batch:, &block)
    end

    def build_chat_template(messages)
      raise Error, 'Model does not provide a chat template' if chat_template.nil? || chat_template.empty?

      count = messages.length
      struct_size = Cpp::LlamaChatMessage.size
      array_ptr = FFI::MemoryPointer.new(struct_size * count)

      messages.each_with_index do |m, i|
        struct_ptr = array_ptr + (i * struct_size)
        msg_struct = Cpp::LlamaChatMessage.new(struct_ptr)
        msg_struct[:role] = FFI::MemoryPointer.from_string(m[:role].to_s)
        msg_struct[:content] = FFI::MemoryPointer.from_string(m[:content].to_s)
      end

      needed = Cpp.llama_chat_apply_template(chat_template, array_ptr, count, true, nil, 0)

      raise Error, 'Failed to apply chat template' if needed < 0

      buf = FFI::MemoryPointer.new(:char, needed)
      written = Cpp.llama_chat_apply_template(chat_template, array_ptr, count, true, buf, needed)

      raise Error, 'Failed to apply chat template' if written < 0

      buf.read_string(written)
    end
  end
end
