# Rllama

Ruby bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp) to run open-source language models locally. Run models like GPT-OSS, Qwen 3, Gemma 3, Llama 3, and many others directly in your Ruby application code.

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'rllama'
```

And then execute:

```bash
bundle install
```

Or install it yourself as:

```bash
gem install rllama
```

## Usage

### Text Generation

Generate text completions using local language models:

```ruby
require 'rllama'

# Load a model
model = Rllama.load_model('lmstudio-community/gemma-3-1B-it-QAT-GGUF/gemma-3-1B-it-QAT-Q4_0.gguf')

# Generate text
result = model.generate('What is the capital of France?')
puts result.text
# => "The capital of France is Paris."

# Access generation statistics
puts "Tokens generated: #{result.stats[:tokens_generated]}"
puts "Tokens per second: #{result.stats[:tps]}"
puts "Duration: #{result.stats[:duration]} seconds"

# Don't forget to close the model when done
model.close
```

#### Generation parameters

Adjust the generation with parameters:

```ruby
result = model.generate(
  'Write a short poem about Ruby programming',
  max_tokens: 2024,
  temperature: 0.8,
  top_k: 40,
  top_p: 0.95,
  min_p: 0.05
)
```

#### Streaming generation

Stream generated text token-by-token:

```ruby
model.generate('Explain quantum computing') do |token|
  print token
end
```

#### System prompt

Include system promt to guide model behavior:

```ruby
result = model.generate(
  'What are best practices for Ruby development?',
  system: 'You are an expert Ruby developer with 10 years of experience.'
)
```

#### Messages list

Pass multiple messages with roles for more complex interactions:

```ruby
result = model.generate([
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'What is the capital of France?' },
  { role: 'assistant', content: 'The capital of France is Paris.' },
  { role: 'user', content: 'What is its population?' }
])
puts result.text
```

### Chat

For ongoing conversations, use a context object that maintains the conversation history:

```ruby
# Initialize a chat context
context = model.init_context

# Send messages and maintain conversation history
response1 = context.message('What is the capital of France?')
puts response1.text
# => "The capital of France is Paris."

response2 = context.message('What is the population of that city?')
puts response2.text
# => "Paris has a population of approximately 2.1 million people..."

response3 = context.message('What was my first message?')
puts response3.text
# => "Your first message was asking about the capital of France."

# The context remembers all previous messages in the conversation

# Close context when done
context.close
```

### Embeddings

Generate vector embeddings for text using embedding models:

```ruby
require 'rllama'

# Load an embedding model
model = Rllama.load_model('lmstudio-community/embeddinggemma-300m-qat-GGUF/embeddinggemma-300m-qat-Q4_0.gguf')

# Generate embedding for a single text
embedding = model.embed('Hello, world!')
puts embedding.length
# => 724 (depending on your model)

# Generate embeddings for multiple sentences
embeddings = model.embed([
  'roses are red',
  'violets are blue',
  'sugar is sweet'
])

puts embeddings.length
# => 3
puts embeddings[0].length
# => 768

model.close
```

#### Vector parameters

By default, embedding vectors are normalized. You can disable normalization with `normalize: false`:

```ruby
# Generate unnormalized embeddings
embedding = model.embed('Sample text', normalize: false)

# Use custom batch size for processing multiple texts
embeddings = model.embed(
  ['roses are red', 'violets are blue', 'sugar is sweet'],
  normalize: true
)
```

## CLI Chat Utility

The `rllama` command-line utility provides an interactive chat interface for conversing with language models. After installing the gem, you can start chatting immediately:

```bash
rllama
```

When you run `rllama` without arguments, it will display:

- **Downloaded models**: Any models you've already downloaded to `~/.rllama/models/`
- **Popular models**: A curated list of popular models available for download, including:
  - Gemma 3 1B
  - Llama 3.2 3B
  - Phi-4
  - Qwen3 30B
  - GPT-OSS
  - And more...

Simply enter the number of the model you want to use. If you select a model that hasn't been downloaded yet, it will be automatically downloaded from Hugging Face.

You can also specify a model path directly:

```bash
rllama path/to/your/model.gguf
```

Once the model loads, you can start chatting.

## Finding Models

You can download GGUF format models from various sources:

- [Hugging Face](https://huggingface.co/models?library=gguf) - Search for models with "GGUF" format

## License

MIT

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/docusealco/rllama.
