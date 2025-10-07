# frozen_string_literal: true

require 'ffi'

module Rllama
  module Cpp
    extend FFI::Library

    LIB_NAME = 'llama'

    PLATFORM =
      case FFI::Platform::OS
      when 'darwin'
        FFI::Platform::ARCH == 'aarch64' ? 'arm64-darwin' : 'x86_64-darwin'
      when 'windows', 'mingw32'
        'x64-mingw32'
      else
        FFI::Platform::ARCH == 'aarch64' ? 'aarch64-linux' : 'x86_64-linux'
      end

    lib_file =
      case FFI::Platform::OS
      when 'darwin'
        "lib#{LIB_NAME}.dylib"
      when 'windows', 'mingw32'
        "#{LIB_NAME}.dll"
      else
        "lib#{LIB_NAME}.so"
      end

    PLATFORM_DIR = File.join(__dir__, PLATFORM)

    platform_path = File.join(PLATFORM_DIR, lib_file)

    lib_paths = []

    lib_paths << platform_path if File.exist?(platform_path)

    ggml_lib_file =
      case FFI::Platform::OS
      when 'darwin'
        'libggml.dylib'
      when 'windows', 'mingw32'
        'ggml.dll'
      else
        'libggml.so'
      end

    ggml_platform_path = File.join(PLATFORM_DIR, ggml_lib_file)
    lib_paths << ggml_platform_path if File.exist?(ggml_platform_path)

    lib_paths +=
      case FFI::Platform::OS
      when 'darwin'
        [
          "lib#{LIB_NAME}.dylib",
          "/opt/homebrew/lib/lib#{LIB_NAME}.dylib",
          "/usr/local/lib/lib#{LIB_NAME}.dylib"
        ]
      when 'windows', 'mingw32'
        [
          "#{LIB_NAME}.dll",
          "lib#{LIB_NAME}.dll"
        ]
      else
        [
          "lib#{LIB_NAME}.so",
          "/usr/lib/lib#{LIB_NAME}.so",
          "/usr/local/lib/lib#{LIB_NAME}.so"
        ]
      end

    ffi_lib lib_paths

    # --- Typedefs and Opaque Pointers ---
    typedef :pointer, :llama_vocab_p
    typedef :pointer, :llama_model_p
    typedef :pointer, :llama_context_p
    typedef :pointer, :llama_sampler_p
    typedef :pointer, :llama_memory_t
    typedef :pointer, :llama_adapter_lora_p
    typedef :pointer, :llama_sampler_context_t
    typedef :pointer, :ggml_threadpool_t

    typedef :int32, :llama_pos
    typedef :int32, :llama_token
    typedef :int32, :llama_seq_id
    typedef :uint32, :llama_state_seq_flags

    # --- Callbacks ---
    # from ggml-backend.h
    callback :ggml_backend_sched_eval_callback, %i[pointer bool pointer], :bool
    callback :ggml_abort_callback, [:pointer], :bool
    callback :ggml_log_callback, %i[int string pointer], :void # Assuming ggml_log_level is int

    # from llama.h
    callback :llama_progress_callback, %i[float pointer], :bool

    # for training
    callback :llama_opt_param_filter, %i[pointer pointer], :bool

    # --- Enums and Constants as Module Constants ---

    # from ggml.h (ggml_type)
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    GGML_TYPE_Q5_0 = 6
    GGML_TYPE_Q5_1 = 7
    GGML_TYPE_Q8_0 = 8
    GGML_TYPE_Q8_1 = 9
    GGML_TYPE_Q2_K = 10
    GGML_TYPE_Q3_K = 11
    GGML_TYPE_Q4_K = 12
    GGML_TYPE_Q5_K = 13
    GGML_TYPE_Q6_K = 14
    GGML_TYPE_Q8_K = 15
    GGML_TYPE_IQ2_XXS = 16
    GGML_TYPE_IQ2_XS = 17
    GGML_TYPE_IQ3_XXS = 18
    GGML_TYPE_IQ1_S = 19
    GGML_TYPE_IQ4_NL = 20
    GGML_TYPE_IQ3_S = 21
    GGML_TYPE_IQ2_S = 22
    GGML_TYPE_IQ4_XS = 23
    GGML_TYPE_I8 = 24
    GGML_TYPE_I16 = 25
    GGML_TYPE_I32 = 26
    GGML_TYPE_I64 = 27
    GGML_TYPE_F64 = 28
    GGML_TYPE_IQ1_M = 29
    GGML_TYPE_COUNT = 30

    # from llama.h
    attach_function :llama_max_devices, [], :size_t
    LLAMA_MAX_DEVICES = llama_max_devices

    LLAMA_DEFAULT_SEED = 0xFFFFFFFF
    LLAMA_TOKEN_NULL = -1
    LLAMA_FILE_MAGIC_GGLA = 0x67676C61
    LLAMA_FILE_MAGIC_GGSN = 0x6767736E
    LLAMA_FILE_MAGIC_GGSQ = 0x67677371
    LLAMA_SESSION_MAGIC = LLAMA_FILE_MAGIC_GGSN
    LLAMA_SESSION_VERSION = 9
    LLAMA_STATE_SEQ_MAGIC = LLAMA_FILE_MAGIC_GGSQ
    LLAMA_STATE_SEQ_VERSION = 2
    LLAMA_STATE_SEQ_FLAGS_SWA_ONLY = 1

    # enum llama_vocab_type
    LLAMA_VOCAB_TYPE_NONE = 0
    LLAMA_VOCAB_TYPE_SPM = 1
    LLAMA_VOCAB_TYPE_BPE = 2
    LLAMA_VOCAB_TYPE_WPM = 3
    LLAMA_VOCAB_TYPE_UGM = 4
    LLAMA_VOCAB_TYPE_RWKV = 5
    LLAMA_VOCAB_TYPE_PLAMO2 = 6

    # enum llama_rope_type
    GGML_ROPE_TYPE_NEOX = 2
    GGML_ROPE_TYPE_MROPE = 8
    GGML_ROPE_TYPE_VISION = 24
    LLAMA_ROPE_TYPE_NONE = -1
    LLAMA_ROPE_TYPE_NORM = 0
    LLAMA_ROPE_TYPE_NEOX = GGML_ROPE_TYPE_NEOX
    LLAMA_ROPE_TYPE_MROPE = GGML_ROPE_TYPE_MROPE
    LLAMA_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION

    # enum llama_token_type
    LLAMA_TOKEN_TYPE_UNDEFINED = 0
    LLAMA_TOKEN_TYPE_NORMAL = 1
    LLAMA_TOKEN_TYPE_UNKNOWN = 2
    LLAMA_TOKEN_TYPE_CONTROL = 3
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4
    LLAMA_TOKEN_TYPE_UNUSED = 5
    LLAMA_TOKEN_TYPE_BYTE = 6

    # enum llama_token_attr
    LLAMA_TOKEN_ATTR_UNDEFINED = 0
    LLAMA_TOKEN_ATTR_UNKNOWN = 1 << 0
    LLAMA_TOKEN_ATTR_UNUSED = 1 << 1
    LLAMA_TOKEN_ATTR_NORMAL = 1 << 2
    LLAMA_TOKEN_ATTR_CONTROL = 1 << 3
    LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4
    LLAMA_TOKEN_ATTR_BYTE = 1 << 5
    LLAMA_TOKEN_ATTR_NORMALIZED = 1 << 6
    LLAMA_TOKEN_ATTR_LSTRIP = 1 << 7
    LLAMA_TOKEN_ATTR_RSTRIP = 1 << 8
    LLAMA_TOKEN_ATTR_SINGLE_WORD = 1 << 9

    # enum llama_ftype
    LLAMA_FTYPE_ALL_F32 = 0
    LLAMA_FTYPE_MOSTLY_F16 = 1
    LLAMA_FTYPE_MOSTLY_Q4_0 = 2
    LLAMA_FTYPE_MOSTLY_Q4_1 = 3
    LLAMA_FTYPE_MOSTLY_Q8_0 = 7
    LLAMA_FTYPE_MOSTLY_Q5_0 = 8
    LLAMA_FTYPE_MOSTLY_Q5_1 = 9
    LLAMA_FTYPE_MOSTLY_Q2_K = 10
    LLAMA_FTYPE_MOSTLY_Q3_K_S = 11
    LLAMA_FTYPE_MOSTLY_Q3_K_M = 12
    LLAMA_FTYPE_MOSTLY_Q3_K_L = 13
    LLAMA_FTYPE_MOSTLY_Q4_K_S = 14
    LLAMA_FTYPE_MOSTLY_Q4_K_M = 15
    LLAMA_FTYPE_MOSTLY_Q5_K_S = 16
    LLAMA_FTYPE_MOSTLY_Q5_K_M = 17
    LLAMA_FTYPE_MOSTLY_Q6_K = 18
    LLAMA_FTYPE_MOSTLY_IQ2_XXS = 19
    LLAMA_FTYPE_MOSTLY_IQ2_XS = 20
    LLAMA_FTYPE_MOSTLY_Q2_K_S = 21
    LLAMA_FTYPE_MOSTLY_IQ3_XS = 22
    LLAMA_FTYPE_MOSTLY_IQ3_XXS = 23
    LLAMA_FTYPE_MOSTLY_IQ1_S = 24
    LLAMA_FTYPE_MOSTLY_IQ4_NL = 25
    LLAMA_FTYPE_MOSTLY_IQ3_S = 26
    LLAMA_FTYPE_MOSTLY_IQ3_M = 27
    LLAMA_FTYPE_MOSTLY_IQ2_S = 28
    LLAMA_FTYPE_MOSTLY_IQ2_M = 29
    LLAMA_FTYPE_MOSTLY_IQ4_XS = 30
    LLAMA_FTYPE_MOSTLY_IQ1_M = 31
    LLAMA_FTYPE_MOSTLY_BF16 = 32
    LLAMA_FTYPE_MOSTLY_TQ1_0 = 36
    LLAMA_FTYPE_MOSTLY_TQ2_0 = 37
    LLAMA_FTYPE_MOSTLY_MXFP4_MOE = 38
    LLAMA_FTYPE_GUESSED = 1024

    # enum llama_rope_scaling_type
    LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1
    LLAMA_ROPE_SCALING_TYPE_NONE = 0
    LLAMA_ROPE_SCALING_TYPE_LINEAR = 1
    LLAMA_ROPE_SCALING_TYPE_YARN = 2
    LLAMA_ROPE_SCALING_TYPE_LONGROPE = 3
    LLAMA_ROPE_SCALING_TYPE_MAX_VALUE = LLAMA_ROPE_SCALING_TYPE_LONGROPE

    # enum llama_pooling_type
    LLAMA_POOLING_TYPE_UNSPECIFIED = -1
    LLAMA_POOLING_TYPE_NONE = 0
    LLAMA_POOLING_TYPE_MEAN = 1
    LLAMA_POOLING_TYPE_CLS = 2
    LLAMA_POOLING_TYPE_LAST = 3
    LLAMA_POOLING_TYPE_RANK = 4

    # enum llama_attention_type
    LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1
    LLAMA_ATTENTION_TYPE_CAUSAL = 0
    LLAMA_ATTENTION_TYPE_NON_CAUSAL = 1

    # enum llama_flash_attn_type
    LLAMA_FLASH_ATTN_TYPE_AUTO = -1
    LLAMA_FLASH_ATTN_TYPE_DISABLED = 0
    LLAMA_FLASH_ATTN_TYPE_ENABLED = 1

    # enum llama_split_mode
    LLAMA_SPLIT_MODE_NONE = 0
    LLAMA_SPLIT_MODE_LAYER = 1
    LLAMA_SPLIT_MODE_ROW = 2

    # enum llama_model_kv_override_type
    LLAMA_KV_OVERRIDE_TYPE_INT = 0
    LLAMA_KV_OVERRIDE_TYPE_FLOAT = 1
    LLAMA_KV_OVERRIDE_TYPE_BOOL = 2
    LLAMA_KV_OVERRIDE_TYPE_STR = 3

    # enum ggml_numa_strategy
    GGML_NUMA_STRATEGY_DISABLED = 0
    GGML_NUMA_STRATEGY_DISTRIBUTE = 1
    GGML_NUMA_STRATEGY_ISOLATE = 2
    GGML_NUMA_STRATEGY_NUMACTL = 3
    GGML_NUMA_STRATEGY_MIRROR = 4
    GGML_NUMA_STRATEGY_COUNT = 5

    # --- Structs and Unions ---

    class LlamaTokenData < FFI::Struct
      layout :id, :llama_token,
             :logit, :float,
             :p, :float
    end

    class LlamaTokenDataArray < FFI::Struct
      layout :data, :pointer, # LlamaTokenData*
             :size, :size_t,
             :selected, :int64,
             :sorted, :bool
    end

    class LlamaBatch < FFI::Struct
      layout :n_tokens, :int32,
             :token, :pointer, # llama_token*
             :embd, :pointer, # float*
             :pos, :pointer, # llama_pos*
             :n_seq_id, :pointer,  # int32*
             :seq_id, :pointer,    # llama_seq_id**
             :logits, :pointer # int8*
    end

    class LlamaModelKvOverrideValue < FFI::Union
      layout :val_i64, :int64,
             :val_f64, :double,
             :val_bool, :bool,
             :val_str, [:char, 128]
    end

    class LlamaModelKvOverride < FFI::Struct
      layout :tag, :int, # enum llama_model_kv_override_type
             :key, [:char, 128],
             :value, LlamaModelKvOverrideValue
    end

    class LlamaModelTensorBuftOverride < FFI::Struct
      layout :pattern, :string,
             :buft, :pointer # ggml_backend_buffer_type_t
    end

    class LlamaModelParams < FFI::Struct
      layout :devices, :pointer, # ggml_backend_dev_t*
             :tensor_buft_overrides, :pointer, # LlamaModelTensorBuftOverride*
             :n_gpu_layers, :int32,
             :split_mode, :int, # enum llama_split_mode
             :main_gpu, :int32,
             :tensor_split, :pointer, # const float *
             :progress_callback, :llama_progress_callback,
             :progress_callback_user_data, :pointer,
             :kv_overrides, :pointer, # const LlamaModelKvOverride*
             :vocab_only, :bool,
             :use_mmap, :bool,
             :use_mlock, :bool,
             :check_tensors, :bool,
             :use_extra_bufts, :bool
    end

    class LlamaContextParams < FFI::Struct
      layout :n_ctx, :uint32,
             :n_batch, :uint32,
             :n_ubatch, :uint32,
             :n_seq_max, :uint32,
             :n_threads, :int32,
             :n_threads_batch, :int32,
             :rope_scaling_type, :int, # enum llama_rope_scaling_type
             :pooling_type, :int,      # enum llama_pooling_type
             :attention_type, :int,    # enum llama_attention_type
             :flash_attn_type, :int, # enum llama_flash_attn_type
             :rope_freq_base, :float,
             :rope_freq_scale, :float,
             :yarn_ext_factor, :float,
             :yarn_attn_factor, :float,
             :yarn_beta_fast, :float,
             :yarn_beta_slow, :float,
             :yarn_orig_ctx, :uint32,
             :defrag_thold, :float,
             :cb_eval, :ggml_backend_sched_eval_callback,
             :cb_eval_user_data, :pointer,
             :type_k, :int, # enum ggml_type
             :type_v, :int, # enum ggml_type
             :abort_callback, :ggml_abort_callback,
             :abort_callback_data, :pointer,
             :embeddings, :bool,
             :offload_kqv, :bool,
             :no_perf, :bool,
             :op_offload, :bool,
             :swa_full, :bool,
             :kv_unified, :bool
    end

    class LlamaModelQuantizeParams < FFI::Struct
      layout :nthread, :int32,
             :ftype, :int, # enum llama_ftype
             :output_tensor_type, :int, # enum ggml_type
             :token_embedding_type, :int, # enum ggml_type
             :allow_requantize, :bool,
             :quantize_output_tensor, :bool,
             :only_copy, :bool,
             :pure, :bool,
             :keep_split, :bool,
             :imatrix, :pointer,
             :kv_overrides, :pointer,
             :tensor_types, :pointer,
             :prune_layers, :pointer
    end

    class LlamaLogitBias < FFI::Struct
      layout :token, :llama_token,
             :bias, :float
    end

    class LlamaSamplerChainParams < FFI::Struct
      layout :no_perf, :bool
    end

    class LlamaChatMessage < FFI::Struct
      layout :role, :pointer,
             :content, :pointer
    end

    class LlamaSamplerI < FFI::Struct; end

    class LlamaSampler < FFI::Struct
      layout :iface, :pointer, # const LlamaSamplerI *
             :ctx, :llama_sampler_context_t
    end

    callback :llama_sampler_i_name, [:pointer], :pointer
    callback :llama_sampler_i_accept, %i[pointer llama_token], :void
    callback :llama_sampler_i_apply, %i[pointer pointer], :void # pointer to LlamaTokenDataArray
    callback :llama_sampler_i_reset, [:pointer], :void
    callback :llama_sampler_i_clone, [:pointer], :pointer
    callback :llama_sampler_i_free, [:pointer], :void

    LlamaSamplerI.layout(
      :name, :llama_sampler_i_name,
      :accept, :llama_sampler_i_accept,
      :apply, :llama_sampler_i_apply,
      :reset, :llama_sampler_i_reset,
      :clone, :llama_sampler_i_clone,
      :free, :llama_sampler_i_free
    )

    class LlamaPerfContextData < FFI::Struct
      layout :t_start_ms, :double,
             :t_load_ms, :double,
             :t_p_eval_ms, :double,
             :t_eval_ms, :double,
             :n_p_eval, :int32,
             :n_eval, :int32,
             :n_reused, :int32
    end

    class LlamaPerfSamplerData < FFI::Struct
      layout :t_sample_ms, :double,
             :n_sample, :int32
    end

    class LlamaOptParams < FFI::Struct
      layout :n_ctx_train, :uint32,
             :param_filter, :llama_opt_param_filter,
             :param_filter_ud, :pointer,
             :get_opt_pars, :pointer, # ggml_opt_get_optimizer_params
             :get_opt_pars_ud, :pointer,
             :optimizer_type, :int # enum ggml_opt_optimizer_type
    end

    # --- Function Attachments ---

    # Default params
    attach_function :llama_model_default_params, [], LlamaModelParams.by_value
    attach_function :llama_context_default_params, [], LlamaContextParams.by_value
    attach_function :llama_sampler_chain_default_params, [], LlamaSamplerChainParams.by_value
    attach_function :llama_model_quantize_default_params, [], LlamaModelQuantizeParams.by_value

    # Backend init/free
    attach_function :llama_backend_init, [], :void
    attach_function :llama_backend_free, [], :void
    attach_function :llama_numa_init, [:int], :void # ggml_numa_strategy
    attach_function :ggml_backend_load_all, [], :void
    attach_function :ggml_backend_load_all_from_path, [:string], :void

    # Threadpool
    attach_function :llama_attach_threadpool, %i[llama_context_p ggml_threadpool_t ggml_threadpool_t], :void
    attach_function :llama_detach_threadpool, [:llama_context_p], :void

    # Model loading
    attach_function :llama_load_model_from_file, [:string, LlamaModelParams.by_value], :llama_model_p # DEPRECATED
    attach_function :llama_model_load_from_file, [:string, LlamaModelParams.by_value], :llama_model_p
    attach_function :llama_model_load_from_splits, [:pointer, :size_t, LlamaModelParams.by_value], :llama_model_p
    attach_function :llama_model_save_to_file, %i[llama_model_p string], :void
    attach_function :llama_free_model, [:llama_model_p], :void # DEPRECATED
    attach_function :llama_model_free, [:llama_model_p], :void

    # Context creation
    attach_function :llama_init_from_model, [:llama_model_p, LlamaContextParams.by_value], :llama_context_p
    # DEPRECATED
    attach_function :llama_new_context_with_model, [:llama_model_p, LlamaContextParams.by_value], :llama_context_p
    attach_function :llama_free, [:llama_context_p], :void

    # System info and support checks
    attach_function :llama_time_us, [], :int64
    # llama_max_devices already attached
    attach_function :llama_max_parallel_sequences, [], :size_t
    attach_function :llama_supports_mmap, [], :bool
    attach_function :llama_supports_mlock, [], :bool
    attach_function :llama_supports_gpu_offload, [], :bool
    attach_function :llama_supports_rpc, [], :bool
    attach_function :llama_flash_attn_type_name, [:int], :string

    # Context info
    attach_function :llama_n_ctx, [:llama_context_p], :uint32
    attach_function :llama_n_batch, [:llama_context_p], :uint32
    attach_function :llama_n_ubatch, [:llama_context_p], :uint32
    attach_function :llama_n_seq_max, [:llama_context_p], :uint32
    attach_function :llama_get_model, [:llama_context_p], :llama_model_p
    attach_function :llama_get_memory, [:llama_context_p], :llama_memory_t
    attach_function :llama_pooling_type, [:llama_context_p], :int # enum llama_pooling_type

    # Model info
    attach_function :llama_model_get_vocab, [:llama_model_p], :llama_vocab_p
    attach_function :llama_model_rope_type, [:llama_model_p], :int # enum llama_rope_type
    attach_function :llama_model_n_ctx_train, [:llama_model_p], :int32
    attach_function :llama_model_n_embd, [:llama_model_p], :int32
    attach_function :llama_model_n_layer, [:llama_model_p], :int32
    attach_function :llama_model_n_head, [:llama_model_p], :int32
    attach_function :llama_model_n_head_kv, [:llama_model_p], :int32
    attach_function :llama_model_n_swa, [:llama_model_p], :int32
    attach_function :llama_model_rope_freq_scale_train, [:llama_model_p], :float
    attach_function :llama_model_n_cls_out, [:llama_model_p], :uint32
    attach_function :llama_model_cls_label, %i[llama_model_p uint32], :string
    attach_function :llama_model_meta_val_str, %i[llama_model_p string pointer size_t], :int32
    attach_function :llama_model_meta_count, [:llama_model_p], :int32
    attach_function :llama_model_meta_key_by_index, %i[llama_model_p int32 pointer size_t], :int32
    attach_function :llama_model_meta_val_str_by_index, %i[llama_model_p int32 pointer size_t], :int32
    attach_function :llama_model_desc, %i[llama_model_p pointer size_t], :int32
    attach_function :llama_model_size, [:llama_model_p], :uint64
    attach_function :llama_model_chat_template, %i[llama_model_p string], :string
    attach_function :llama_model_n_params, [:llama_model_p], :uint64
    attach_function :llama_model_has_encoder, [:llama_model_p], :bool
    attach_function :llama_model_has_decoder, [:llama_model_p], :bool
    attach_function :llama_model_decoder_start_token, [:llama_model_p], :llama_token
    attach_function :llama_model_is_recurrent, [:llama_model_p], :bool
    attach_function :llama_model_is_diffusion, [:llama_model_p], :bool

    # Vocab info
    attach_function :llama_vocab_type, [:llama_vocab_p], :int # enum llama_vocab_type
    attach_function :llama_vocab_n_tokens, [:llama_vocab_p], :int32

    # Quantization
    attach_function :llama_model_quantize, %i[string string pointer], :uint32

    # Adapters
    attach_function :llama_adapter_lora_init, %i[llama_model_p string], :llama_adapter_lora_p
    attach_function :llama_adapter_meta_val_str, %i[llama_adapter_lora_p string pointer size_t], :int32
    attach_function :llama_adapter_meta_count, [:llama_adapter_lora_p], :int32
    attach_function :llama_adapter_meta_key_by_index, %i[llama_adapter_lora_p int32 pointer size_t], :int32
    attach_function :llama_adapter_meta_val_str_by_index, %i[llama_adapter_lora_p int32 pointer size_t], :int32
    attach_function :llama_adapter_lora_free, [:llama_adapter_lora_p], :void
    attach_function :llama_adapter_get_alora_n_invocation_tokens, [:llama_adapter_lora_p], :uint64
    attach_function :llama_adapter_get_alora_invocation_tokens, [:llama_adapter_lora_p], :pointer # const llama_token*
    attach_function :llama_set_adapter_lora, %i[llama_context_p llama_adapter_lora_p float], :int32
    attach_function :llama_rm_adapter_lora, %i[llama_context_p llama_adapter_lora_p], :int32
    attach_function :llama_clear_adapter_lora, [:llama_context_p], :void
    attach_function :llama_apply_adapter_cvec, %i[llama_context_p pointer size_t int32 int32 int32], :int32

    # Memory management
    attach_function :llama_memory_clear, %i[llama_memory_t bool], :void
    attach_function :llama_memory_seq_rm, %i[llama_memory_t llama_seq_id llama_pos llama_pos], :bool
    attach_function :llama_memory_seq_cp, %i[llama_memory_t llama_seq_id llama_seq_id llama_pos llama_pos], :void
    attach_function :llama_memory_seq_keep, %i[llama_memory_t llama_seq_id], :void
    attach_function :llama_memory_seq_add, %i[llama_memory_t llama_seq_id llama_pos llama_pos llama_pos], :void
    attach_function :llama_memory_seq_div, %i[llama_memory_t llama_seq_id llama_pos llama_pos int], :void
    attach_function :llama_memory_seq_pos_min, %i[llama_memory_t llama_seq_id], :llama_pos
    attach_function :llama_memory_seq_pos_max, %i[llama_memory_t llama_seq_id], :llama_pos
    attach_function :llama_memory_can_shift, [:llama_memory_t], :bool

    # State / sessions
    attach_function :llama_state_get_size, [:llama_context_p], :size_t
    attach_function :llama_state_get_data, %i[llama_context_p pointer size_t], :size_t
    attach_function :llama_state_set_data, %i[llama_context_p pointer size_t], :size_t
    attach_function :llama_state_load_file, %i[llama_context_p string pointer size_t pointer], :bool
    attach_function :llama_state_save_file, %i[llama_context_p string pointer size_t], :bool
    attach_function :llama_state_seq_get_size, %i[llama_context_p llama_seq_id], :size_t
    attach_function :llama_state_seq_get_data, %i[llama_context_p pointer size_t llama_seq_id], :size_t
    attach_function :llama_state_seq_set_data, %i[llama_context_p pointer size_t llama_seq_id], :size_t
    attach_function :llama_state_seq_save_file, %i[llama_context_p string llama_seq_id pointer size_t], :size_t
    attach_function :llama_state_seq_load_file, %i[llama_context_p string llama_seq_id pointer size_t pointer], :size_t
    attach_function :llama_state_seq_get_size_ext, %i[llama_context_p llama_seq_id llama_state_seq_flags], :size_t
    attach_function :llama_state_seq_get_data_ext,
                    %i[llama_context_p pointer size_t llama_seq_id llama_state_seq_flags], :size_t
    attach_function :llama_state_seq_set_data_ext,
                    %i[llama_context_p pointer size_t llama_seq_id llama_state_seq_flags], :size_t

    # Decoding
    attach_function :llama_batch_get_one, %i[pointer int32], LlamaBatch.by_value
    attach_function :llama_batch_init, %i[int32 int32 int32], LlamaBatch.by_value
    attach_function :llama_batch_free, [LlamaBatch.by_value], :void
    attach_function :llama_encode, [:llama_context_p, LlamaBatch.by_value], :int32
    attach_function :llama_decode, [:llama_context_p, LlamaBatch.by_value], :int32

    # Decoding settings
    attach_function :llama_set_n_threads, %i[llama_context_p int32 int32], :void
    attach_function :llama_n_threads, [:llama_context_p], :int32
    attach_function :llama_n_threads_batch, [:llama_context_p], :int32
    attach_function :llama_set_embeddings, %i[llama_context_p bool], :void
    attach_function :llama_set_causal_attn, %i[llama_context_p bool], :void
    attach_function :llama_set_warmup, %i[llama_context_p bool], :void
    attach_function :llama_set_abort_callback, %i[llama_context_p ggml_abort_callback pointer], :void
    attach_function :llama_synchronize, [:llama_context_p], :void

    # Get results
    attach_function :llama_get_logits, [:llama_context_p], :pointer # float*
    attach_function :llama_get_logits_ith, %i[llama_context_p int32], :pointer # float*
    attach_function :llama_get_embeddings, [:llama_context_p], :pointer # float*
    attach_function :llama_get_embeddings_ith, %i[llama_context_p int32], :pointer # float*
    attach_function :llama_get_embeddings_seq, %i[llama_context_p llama_seq_id], :pointer # float*

    # Vocab utils
    attach_function :llama_vocab_get_text, %i[llama_vocab_p llama_token], :string
    attach_function :llama_vocab_get_score, %i[llama_vocab_p llama_token], :float
    attach_function :llama_vocab_get_attr, %i[llama_vocab_p llama_token], :int # enum llama_token_attr
    attach_function :llama_vocab_is_eog, %i[llama_vocab_p llama_token], :bool
    attach_function :llama_vocab_is_control, %i[llama_vocab_p llama_token], :bool
    attach_function :llama_vocab_bos, [:llama_vocab_p], :llama_token
    attach_function :llama_vocab_eos, [:llama_vocab_p], :llama_token
    attach_function :llama_vocab_eot, [:llama_vocab_p], :llama_token
    attach_function :llama_vocab_sep, [:llama_vocab_p], :llama_token
    attach_function :llama_vocab_nl, [:llama_vocab_p], :llama_token
    attach_function :llama_vocab_pad, [:llama_vocab_p], :llama_token
    attach_function :llama_vocab_mask, [:llama_vocab_p], :llama_token
    attach_function :llama_vocab_get_add_bos, [:llama_vocab_p], :bool
    attach_function :llama_vocab_get_add_eos, [:llama_vocab_p], :bool
    attach_function :llama_vocab_get_add_sep, [:llama_vocab_p], :bool
    attach_function :llama_vocab_fim_pre, [:llama_vocab_p], :llama_token
    attach_function :llama_vocab_fim_suf, [:llama_vocab_p], :llama_token
    attach_function :llama_vocab_fim_mid, [:llama_vocab_p], :llama_token
    attach_function :llama_vocab_fim_pad, [:llama_vocab_p], :llama_token
    attach_function :llama_vocab_fim_rep, [:llama_vocab_p], :llama_token
    attach_function :llama_vocab_fim_sep, [:llama_vocab_p], :llama_token
    attach_function :llama_vocab_cls, [:llama_vocab_p], :llama_token # DEPRECATED

    # Tokenization
    attach_function :llama_tokenize, %i[llama_vocab_p string int32 pointer int32 bool bool], :int32
    attach_function :llama_token_to_piece, %i[llama_vocab_p llama_token pointer int32 int32 bool], :int32
    attach_function :llama_detokenize, %i[llama_vocab_p pointer int32 pointer int32 bool bool], :int32

    # Chat templates
    attach_function :llama_chat_apply_template, %i[string pointer size_t bool pointer int32], :int32
    attach_function :llama_chat_builtin_templates, %i[pointer size_t], :int32

    # Sampling API
    attach_function :llama_sampler_init, %i[pointer llama_sampler_context_t], :llama_sampler_p
    attach_function :llama_sampler_name, [:llama_sampler_p], :string
    attach_function :llama_sampler_accept, %i[llama_sampler_p llama_token], :void
    attach_function :llama_sampler_apply, %i[llama_sampler_p pointer], :void
    attach_function :llama_sampler_reset, [:llama_sampler_p], :void
    attach_function :llama_sampler_clone, [:llama_sampler_p], :llama_sampler_p
    attach_function :llama_sampler_free, [:llama_sampler_p], :void

    # Sampler chain
    attach_function :llama_sampler_chain_init, [LlamaSamplerChainParams.by_value], :llama_sampler_p
    attach_function :llama_sampler_chain_add, %i[llama_sampler_p llama_sampler_p], :void
    attach_function :llama_sampler_chain_get, %i[llama_sampler_p int32], :llama_sampler_p
    attach_function :llama_sampler_chain_n, [:llama_sampler_p], :int
    attach_function :llama_sampler_chain_remove, %i[llama_sampler_p int32], :llama_sampler_p

    # Built-in samplers
    attach_function :llama_sampler_init_greedy, [], :llama_sampler_p
    attach_function :llama_sampler_init_dist, [:uint32], :llama_sampler_p
    attach_function :llama_sampler_init_top_k, [:int32], :llama_sampler_p
    attach_function :llama_sampler_init_top_p, %i[float size_t], :llama_sampler_p
    attach_function :llama_sampler_init_min_p, %i[float size_t], :llama_sampler_p
    attach_function :llama_sampler_init_typical, %i[float size_t], :llama_sampler_p
    attach_function :llama_sampler_init_temp, [:float], :llama_sampler_p
    attach_function :llama_sampler_init_temp_ext, %i[float float float], :llama_sampler_p
    attach_function :llama_sampler_init_xtc, %i[float float size_t uint32], :llama_sampler_p
    attach_function :llama_sampler_init_top_n_sigma, [:float], :llama_sampler_p
    attach_function :llama_sampler_init_mirostat, %i[int32 uint32 float float int32], :llama_sampler_p
    attach_function :llama_sampler_init_mirostat_v2, %i[uint32 float float], :llama_sampler_p
    attach_function :llama_sampler_init_grammar, %i[llama_vocab_p string string], :llama_sampler_p
    attach_function :llama_sampler_init_grammar_lazy_patterns,
                    %i[llama_vocab_p string string pointer size_t pointer size_t], :llama_sampler_p
    attach_function :llama_sampler_init_penalties, %i[int32 float float float], :llama_sampler_p
    attach_function :llama_sampler_init_dry, %i[llama_vocab_p int32 float float int32 int32 pointer size_t],
                    :llama_sampler_p
    attach_function :llama_sampler_init_logit_bias, %i[int32 int32 pointer], :llama_sampler_p
    attach_function :llama_sampler_init_infill, [:llama_vocab_p], :llama_sampler_p
    attach_function :llama_sampler_get_seed, [:llama_sampler_p], :uint32
    attach_function :llama_sampler_sample, %i[llama_sampler_p llama_context_p int32], :llama_token

    # Model split
    attach_function :llama_split_path, %i[pointer size_t string int int], :int
    attach_function :llama_split_prefix, %i[pointer size_t string int int], :int

    # Logging
    attach_function :llama_print_system_info, [], :string
    attach_function :llama_log_set, %i[ggml_log_callback pointer], :void

    # Performance utils
    attach_function :llama_perf_context, [:llama_context_p], LlamaPerfContextData.by_value
    attach_function :llama_perf_context_print, [:llama_context_p], :void
    attach_function :llama_perf_context_reset, [:llama_context_p], :void
    attach_function :llama_perf_sampler, [:llama_sampler_p], LlamaPerfSamplerData.by_value
    attach_function :llama_perf_sampler_print, [:llama_sampler_p], :void
    attach_function :llama_perf_sampler_reset, [:llama_sampler_p], :void

    # Training
    attach_function :llama_opt_param_filter_all, %i[pointer pointer], :bool
    attach_function :llama_opt_init, [:llama_context_p, :llama_model_p, LlamaOptParams.by_value], :void
    attach_function :llama_opt_epoch, %i[llama_context_p pointer pointer pointer int64 pointer pointer], :void

    SILENCE_LOG_CALLBACK = FFI::Function.new(:void, %i[int string pointer], proc {})

    module_function

    def silence_log!
      llama_log_set(SILENCE_LOG_CALLBACK, nil)
    end

    def set_log(io = $stdout)
      @log_callback = FFI::Function.new(:void, %i[int string pointer]) { |_level, msg, _ud| io << msg }

      llama_log_set(@log_callback, nil)
    end

    silence_log!

    if File.directory?(PLATFORM_DIR)
      ggml_backend_load_all_from_path(PLATFORM_DIR)
    else
      ggml_backend_load_all
    end

    freeze
  end
end
