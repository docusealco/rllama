# frozen_string_literal: true

require 'readline'

module Rllama
  class Cli
    POPULAR_MODELS = [
      { path: 'lmstudio-community/gemma-3-1B-it-QAT-GGUF/gemma-3-1B-it-QAT-Q4_0.gguf', size: 720_425_472 },
      { path: 'lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf', size: 12_109_565_632 },
      { path: 'bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf', size: 2_019_377_696 },
      { path: 'unsloth/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q3_K_S.gguf', size: 13_292_468_800 },
      { path: 'inclusionAI/Ling-mini-2.0-GGUF/Ling-mini-2.0-Q4_K_M.gguf', size: 9_911_575_072 },
      { path: 'unsloth/gemma-3n-E4B-it-GGUF/gemma-3n-E4B-it-Q4_K_S.gguf', size: 4_404_697_216 },
      { path: 'microsoft/phi-4-gguf/phi-4-Q4_K_S.gguf', size: 8_440_762_560 }
    ].freeze

    COLOR_CODES = {
      red: 31,
      green: 32,
      yellow: 33,
      blue: 34,
      magenta: 35,
      cyan: 36
    }.freeze

    def self.start(args)
      new(args).run
    end

    def initialize(args)
      @args = args

      @model_path = args.first
    end

    def run
      model_path = select_or_load_model

      puts "\n#{colorize('Loading model...', :yellow)}"

      model = Rllama.load_model(model_path)
      context = model.init_context

      puts colorize('Model loaded successfully!', :green)
      puts "\n#{colorize('Chat started. Type your message and press Enter. Type "exit" or "quit" to end the chat.',
                         :cyan)}\n\n"

      chat_loop(context)
    rescue Interrupt
      puts "\n\n#{colorize('Chat interrupted. Goodbye!', :yellow)}"
      exit(0)
    rescue StandardError => e
      puts "\n#{colorize("Error: #{e.message}", :red)}"
      exit(1)
    ensure
      context&.close
      model&.close
    end

    private

    def select_or_load_model
      return @model_path if @model_path

      downloaded_models = find_downloaded_models
      downloaded_model_names = downloaded_models.map { |path| File.basename(path, '.gguf') }

      available_popular = POPULAR_MODELS.reject do |popular_model|
        popular_filename = File.basename(popular_model[:path], '.gguf')
        downloaded_model_names.any?(popular_filename)
      end

      all_choices = []
      current_index = 1

      unless downloaded_models.empty?
        puts "#{colorize('Downloaded models:', :cyan)}\n\n"

        downloaded_models.each do |model|
          display_name = File.basename(model, '.gguf')
          size = format_file_size(File.size(model))
          puts "  #{colorize(current_index.to_s, :green)}. #{display_name} #{colorize("(#{size})", :yellow)}"
          all_choices << model
          current_index += 1
        end

        puts "\n"
      end

      unless available_popular.empty?
        puts "#{colorize('Popular models (not downloaded):', :cyan)}\n\n"

        available_popular.each do |model|
          display_name = File.basename(model[:path], '.gguf')
          puts "  #{colorize(current_index.to_s, :green)}. " \
               "#{display_name} #{colorize("(#{format_file_size(model[:size])})", :yellow)}"
          all_choices << model[:path]
          current_index += 1
        end

        puts "\n"
      end

      if all_choices.empty?
        puts colorize('No models available', :yellow)
        exit(1)
      end

      print colorize("Enter number (1-#{all_choices.length}): ", :cyan)

      choice = $stdin.gets&.strip.to_i

      if choice < 1 || choice > all_choices.length
        puts colorize('Invalid choice', :red)

        exit(1)
      end

      all_choices[choice - 1]
    end

    def find_downloaded_models
      models_dir = File.join(Dir.home, '.rllama', 'models')

      return [] unless Dir.exist?(models_dir)

      Dir.glob(File.join(models_dir, '**', '*.gguf')).reject do |path|
        basename = File.basename(path)

        basename.start_with?('~', '!')
      end
    end

    def format_file_size(bytes)
      gb = bytes / (1024.0**3)

      if gb >= 1.0
        format('%.1fGB', gb)
      else
        mb = bytes / (1024.0**2)

        format('%dMB', mb.round)
      end
    end

    def chat_loop(context)
      loop do
        user_input = Readline.readline('> ', false)&.strip

        break if user_input.nil?

        next if user_input.empty?

        if user_input.downcase == 'exit' || user_input.downcase == 'quit'
          puts "\n#{colorize('Goodbye!', :yellow)}"

          break
        end

        puts "\n"

        print "#{colorize('Assistant:', :magenta, bold: true)} "

        context.generate(user_input) do |token|
          print token
          $stdout.flush
        end

        puts "\n\n"
      end
    end

    def colorize(text, color, bold: false)
      return text unless $stdout.tty?

      code = COLOR_CODES[color] || 37

      prefix = bold ? "\e[1;#{code}m" : "\e[#{code}m"

      "#{prefix}#{text}\e[0m"
    end
  end
end
