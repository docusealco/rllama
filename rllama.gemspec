# frozen_string_literal: true

require_relative 'lib/rllama/version'

Gem::Specification.new do |spec|
  spec.name = 'rllama'
  spec.version = Rllama::VERSION
  spec.authors = ['Pete Matsyburka']
  spec.email = ['pete@docuseal.com']
  spec.summary = 'Ruby bindings for llama.cpp to run local LLMs with Ruby.'
  spec.description = 'Ruby bindings for Llama.cpp to run local LLMs in Ruby applications.'
  spec.license = 'MIT'
  spec.required_ruby_version = '>= 3.1.0'

  spec.metadata = {
    'bug_tracker_uri' => 'https://github.com/docusealco/rllama/issues',
    'homepage_uri' => 'https://github.com/docusealco/rllama',
    'source_code_uri' => 'https://github.com/docusealco/rllama',
    'rubygems_mfa_required' => 'true'
  }

  # Base files included in all gems
  base_files = Dir[
    'lib/**/*.rb',
    'LICENSE',
    'README.md'
  ]

  # If building for a specific platform, include the binary
  spec.files =
    if spec.platform.to_s == 'ruby'
      base_files
    else
      platform_dir = case spec.platform.to_s
                     when /x86_64-linux/
                       'x86_64-linux'
                     when /aarch64-linux/
                       'aarch64-linux'
                     when /x86_64-darwin/
                       'x86_64-darwin'
                     when /arm64-darwin/
                       'arm64-darwin'
                     when /x64-mingw32/
                       'x64-mingw32'
                     when /x64-mingw-ucrt/
                       'x64-mingw-ucrt'
                     end

      if platform_dir
        base_files + Dir["lib/rllama/#{platform_dir}/*"]
      else
        base_files
      end
    end

  spec.require_paths = ['lib']
  spec.bindir = 'bin'
  spec.executables = ['rllama']

  spec.add_dependency 'ffi', '>= 1.0'
end
