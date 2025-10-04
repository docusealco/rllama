module Rllama
  autoload :Model, 'rllama/model'
  autoload :Context, 'rllama/context'
  autoload :Cpp, 'rllama/cpp'
  autoload :VERSION, 'rllama/version'

  Result = Struct.new(:text, :stats, keyword_init: true)
  Error = Class.new(StandardError)

  module_function

  def load_model(path_or_name)
    model = Model.new(path_or_name)

    if block_given?
      begin
        return yield model
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
