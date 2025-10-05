# frozen_string_literal: true

module Rllama
  autoload :Model, 'rllama/model'
  autoload :Loader, 'rllama/loader'
  autoload :Context, 'rllama/context'
  autoload :Cpp, 'rllama/cpp'
  autoload :Cli, 'rllama/cli'
  autoload :VERSION, 'rllama/version'

  Result = Struct.new(:text, :stats, keyword_init: true)
  Error = Class.new(StandardError)

  module_function

  def load_model(path_or_name, dir: nil)
    model = Model.new(path_or_name, dir:)

    if block_given?
      begin
        yield model
      ensure
        model.close
      end
    else
      model
    end
  end

  def silence_log!
    Cpp.silence_log!
  end

  def set_log(io = $stdout)
    Cpp.set_log(io)
  end
end
