# frozen_string_literal: true

require 'ffi'
require 'json'

module Rllama
  module Common
    extend FFI::Library

    REASONING_FORMAT_AUTO = 1

    lib_file =
      case FFI::Platform::OS
      when 'darwin'
        'libllama_common.dylib'
      when 'windows', 'mingw32'
        'llama_common.dll'
      else
        'libllama_common.so'
      end

    ffi_lib File.join(Cpp::PLATFORM_DIR, lib_file)

    attach_function :rlc_free, [:pointer], :void
    attach_function :rlc_chat_templates_init, %i[pointer string string string pointer], :pointer
    attach_function :rlc_chat_templates_free, [:pointer], :void
    attach_function :rlc_chat_apply, %i[pointer string pointer], :pointer
    attach_function :rlc_chat_parse, %i[string bool string pointer], :pointer
    attach_function :rlc_sampler_init, %i[pointer string pointer], :pointer
    attach_function :rlc_sampler_free, [:pointer], :void
    attach_function :rlc_sampler_sample, %i[pointer pointer int32], :int32
    attach_function :rlc_sampler_accept, %i[pointer int32 bool], :void
    attach_function :rlc_sampler_reset, [:pointer], :void
    attach_function :rlc_sampler_print, [:pointer], :pointer

    module_function

    def chat_templates_init(model, source, bos_token, eos_token)
      with_err do |err|
        handle = rlc_chat_templates_init(model, source, bos_token, eos_token, err)

        raise Error, "Failed to parse chat template: #{read_err(err)}" if handle.null?

        handle
      end
    end

    def chat_templates_free(handle)
      rlc_chat_templates_free(handle)
    end

    def chat_apply(handle, messages:, tools: nil, add_generation_prompt: true, enable_thinking: false,
                   add_bos: false, add_eos: false)
      inputs = { messages:, tools:, add_generation_prompt:, enable_thinking:, add_bos:, add_eos:,
                 reasoning_format: enable_thinking ? REASONING_FORMAT_AUTO : 0 }.to_json

      with_err do |err|
        result = rlc_chat_apply(handle, inputs, err)

        raise Error, "Failed to apply chat template: #{read_err(err)}" if result.null?

        JSON.parse(take(result))
      end
    end

    def chat_parse(text, params, is_partial: false, reasoning: false)
      parse_params = params.slice('format', 'parser', 'generation_prompt')
      parse_params['reasoning_format'] = REASONING_FORMAT_AUTO if reasoning
      params_json = parse_params.to_json

      with_err do |err|
        result = rlc_chat_parse(text, is_partial, params_json, err)

        raise Error, "Failed to parse output: #{read_err(err)}" if result.null?

        JSON.parse(take(result))
      end
    end

    def sampler_init(model, params)
      with_err do |err|
        handle = rlc_sampler_init(model, params.to_json, err)

        raise Error, "Failed to initialize sampler: #{read_err(err)}" if handle.null?

        handle
      end
    end

    def sampler_free(handle)
      rlc_sampler_free(handle)
    end

    def sampler_sample(handle, ctx, idx = -1)
      rlc_sampler_sample(handle, ctx, idx)
    end

    def sampler_accept(handle, token, accept_grammar: true)
      rlc_sampler_accept(handle, token, accept_grammar)
    end

    def sampler_reset(handle)
      rlc_sampler_reset(handle)
    end

    def sampler_print(handle)
      take(rlc_sampler_print(handle))
    end

    def with_err
      yield FFI::MemoryPointer.new(:pointer)
    end

    def take(ptr)
      str = ptr.read_string

      rlc_free(ptr)

      str
    end

    def read_err(err)
      msg_ptr = err.read_pointer

      return 'unknown error' if msg_ptr.null?

      msg = msg_ptr.read_string

      rlc_free(msg_ptr)

      msg
    end
  end
end
