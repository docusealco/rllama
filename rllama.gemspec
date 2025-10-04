# frozen_string_literal: true

require_relative 'lib/rllama/version'

Gem::Specification.new do |spec|
  spec.name = 'rllama'
  spec.version = Rllama::VERSION
  spec.authors = ['Pete Matsyburka']
  spec.email = ['pete@docuseal.com']
  spec.summary = 'Ruby bindings for Llama API'
  spec.description = 'Ruby bindings for Llama.cpp to run local LLMs in Ruby applications.'
  spec.license = 'MIT'
  spec.required_ruby_version = '>= 3.0.0'

  spec.metadata = {
    'bug_tracker_uri' => 'https://github.com/docusealco/rllama/issues',
    'homepage_uri' => 'https://github.com/docusealco/rllama',
    'source_code_uri' => 'https://github.com/docusealco/rllama',
    'rubygems_mfa_required' => 'true'
  }

  spec.files = Dir[
    'lib/**/*',
    'LICENSE',
    'README.md'
  ]

  spec.require_paths = ['lib']

  spec.add_dependency 'ffi', '>= 1.0'
end
