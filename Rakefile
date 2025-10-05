# frozen_string_literal: true

require 'bundler/gem_helper'
require 'fileutils'
require 'net/http'
require 'uri'
require 'zip'

LLAMA_CPP_VERSION = ENV.fetch('LLAMA_CPP_VERSION', 'b6691')

PLATFORMS = {
  'x86_64-linux' => {
    gem_platform: 'x86_64-linux-gnu',
    asset_name: 'llama-%<version>s-bin-ubuntu-x64.zip',
    lib_extension: '.so'
  },
  'aarch64-linux' => {
    gem_platform: 'aarch64-linux-gnu',
    lib_extension: '.so'
  },
  'aarch64-linux-musl' => {
    gem_platform: 'aarch64-linux-musl',
    lib_extension: '.so'
  },
  'x86_64-darwin' => {
    gem_platform: 'x86_64-darwin',
    asset_name: 'llama-%<version>s-bin-macos-x64.zip',
    lib_extension: '.dylib'
  },
  'arm64-darwin' => {
    gem_platform: 'arm64-darwin',
    asset_name: 'llama-%<version>s-bin-macos-arm64.zip',
    lib_extension: '.dylib'
  },
  'x64-mingw32' => {
    gem_platform: 'x64-mingw32',
    asset_name: 'llama-%<version>s-bin-win-cpu-x64.zip',
    lib_extension: '.dll'
  },
  'x64-mingw-ucrt' => {
    gem_platform: 'x64-mingw-ucrt',
    asset_name: 'llama-%<version>s-bin-win-cpu-x64.zip',
    lib_extension: '.dll'
  }
}.freeze

namespace :llama do
  desc 'Download llama.cpp binaries for all platforms'
  task :download_all do
    PLATFORMS.each_key do |platform|
      Rake::Task['llama:download'].execute(Rake::TaskArguments.new([:platform], [platform]))
    end
  end

  desc 'Download llama.cpp binaries for a specific platform'
  task :download, [:platform] do |_t, args|
    platform = args[:platform] || detect_platform
    download_llama_binary(platform)
  end

  desc 'Clean downloaded binaries'
  task :clean do
    FileUtils.rm_rf('tmp/llama')
    PLATFORMS.each_key do |platform|
      FileUtils.rm_rf("lib/rllama/#{platform}")
    end
  end

  desc 'Build ARM64 Linux binaries using Docker'
  task :build_arm do
    puts 'Building llama.cpp Docker image for ARM64...'
    sh 'docker build -t llamacpp-builder-arm -f Dockerfile.llamacpp-arm .'

    puts 'Creating container from image...'
    container_id = `docker create llamacpp-builder-arm`.strip
    puts "Container ID: #{container_id}"

    begin
      puts 'Creating output directory...'
      FileUtils.mkdir_p('build/linux-aarch64')

      puts 'Copying built files from container to build/linux-aarch64...'
      sh "docker cp #{container_id}:/workspace/llama.cpp/build/. build/linux-aarch64/"

      puts 'Build complete! Files are available in build/linux-aarch64/'
      sh 'ls -lh build/linux-aarch64/'
    ensure
      puts 'Cleaning up container...'
      sh "docker rm #{container_id}"
    end
  end

  desc 'Build ARM64 Linux musl binaries using Docker'
  task :build_arm_musl do
    puts 'Building llama.cpp Docker image for ARM64 musl...'
    sh 'docker build -t llamacpp-builder-arm-musl -f Dockerfile.llamacpp-arm-musl .'

    puts 'Creating container from image...'
    container_id = `docker create llamacpp-builder-arm-musl`.strip
    puts "Container ID: #{container_id}"

    begin
      puts 'Creating output directory...'
      FileUtils.mkdir_p('build/linux-aarch64-musl')

      puts 'Copying built files from container to build/linux-aarch64-musl...'
      sh "docker cp #{container_id}:/workspace/llama.cpp/build/. build/linux-aarch64-musl/"

      puts 'Build complete! Files are available in build/linux-aarch64-musl/'
      sh 'ls -lh build/linux-aarch64-musl/'
    ensure
      puts 'Cleaning up container...'
      sh "docker rm #{container_id}"
    end
  end

  desc 'Build ARM64 Linux and copy to lib directory'
  task build_and_install_arm: ['llama:build_arm'] do
    puts "\nCopying ARM64 binaries to lib directory..."
    download_llama_binary('aarch64-linux')
    puts 'ARM64 Linux binaries ready!'
  end

  desc 'Build ARM64 Linux musl and copy to lib directory'
  task build_and_install_arm_musl: ['llama:build_arm_musl'] do
    puts "\nCopying ARM64 musl binaries to lib directory..."
    download_llama_binary('aarch64-linux-musl')
    puts 'ARM64 Linux musl binaries ready!'
  end
end

namespace :gem do
  desc 'Build platform-specific gems'
  task build_all: ['llama:download_all'] do
    PLATFORMS.each_key do |platform|
      Rake::Task['gem:build_platform'].execute(Rake::TaskArguments.new([:platform], [platform]))
    end
  end

  desc 'Build gem for a specific platform'
  task :build_platform, [:platform] do |_t, args|
    platform = args[:platform]
    raise ArgumentError, "Unknown platform: #{platform}" unless PLATFORMS.key?(platform)

    build_platform_gem(platform)
  end

  desc 'Build base gem (ruby platform)'
  task build: :build_base

  task :build_base do
    sh 'gem build rllama.gemspec'
  end

  desc 'Release all platform gems'
  task release_all: ['gem:build_all'] do
    Dir.glob('pkg/*.gem').each do |gem_file|
      sh "gem push #{gem_file}"
    end
  end
end

namespace :build do
  desc 'Download binaries for specific platforms'
  task :download, [:platforms] do |_t, args|
    platforms = parse_platforms(args[:platforms])
    puts "Downloading binaries for #{platforms.length} platforms..."

    platforms.each do |platform|
      puts "\n==> Downloading #{platform}..."
      Rake::Task['llama:download'].execute(Rake::TaskArguments.new([:platform], [platform]))
    end
  end

  desc 'Build all gems (download and build)'
  task :all, [:platforms] do |_t, args|
    platforms = parse_platforms(args[:platforms])
    puts 'Building all platform gems...'

    # Download binaries
    puts "\nDownloading binaries for #{platforms.length} platforms..."
    platforms.each do |platform|
      puts "\n==> Downloading #{platform}..."
      Rake::Task['llama:download'].execute(Rake::TaskArguments.new([:platform], [platform]))
    end

    # Build base gem
    puts "\n==> Building base gem..."
    Rake::Task['gem:build_base'].invoke

    # Build platform gems
    puts "\nBuilding gems for #{platforms.length} platforms..."
    platforms.each do |platform|
      puts "\n==> Building #{platform} gem..."
      Rake::Task['gem:build_platform'].execute(Rake::TaskArguments.new([:platform], [platform]))
    end

    puts "\nBuild complete! Gems are in pkg/"
    list_built_gems
  end

  desc 'Build gems only (assumes binaries already downloaded)'
  task :gems, [:platforms] do |_t, args|
    platforms = parse_platforms(args[:platforms])
    puts "\nBuilding gems for #{platforms.length} platforms..."

    # Build base gem
    puts "\n==> Building base gem..."
    Rake::Task['gem:build_base'].invoke

    # Build platform gems
    platforms.each do |platform|
      puts "\n==> Building #{platform} gem..."
      Rake::Task['gem:build_platform'].execute(Rake::TaskArguments.new([:platform], [platform]))
    end

    puts "\nBuild complete! Gems are in pkg/"
    list_built_gems
  end
end

namespace :release do
  desc 'Release all gems to RubyGems'
  task :all do
    puts "\nReleasing all gems..."

    gems = Dir.glob('pkg/*.gem')
    abort('No gems found in pkg/. Run rake build:all first.') if gems.empty?

    gems.each do |gem|
      puts "\n==> Pushing #{File.basename(gem)}..."
      sh "gem push #{gem}"
    end

    puts "\nAll gems released successfully!"
  end

  desc 'List built gems'
  task :list do
    list_built_gems
  end
end

def detect_platform
  case RbConfig::CONFIG['host_os']
  when /darwin/
    RbConfig::CONFIG['host_cpu'] == 'arm64' ? 'arm64-darwin' : 'x86_64-darwin'
  when /linux/
    RbConfig::CONFIG['host_cpu'] == 'aarch64' ? 'aarch64-linux' : 'x86_64-linux'
  when /mingw|mswin/
    'x64-mingw32'
  else
    raise "Unsupported platform: #{RbConfig::CONFIG['host_os']}"
  end
end

def download_llama_binary(platform)
  config = PLATFORMS[platform]
  raise ArgumentError, "Unknown platform: #{platform}" unless config

  # Special case: use local build for aarch64-linux and aarch64-linux-musl
  if platform == 'aarch64-linux' || platform == 'aarch64-linux-musl'
    copy_local_build(platform, config)
    return
  end

  asset_name = format(config[:asset_name], version: LLAMA_CPP_VERSION)
  download_url = "https://github.com/ggml-org/llama.cpp/releases/download/#{LLAMA_CPP_VERSION}/#{asset_name}"

  tmp_dir = 'tmp/llama'
  FileUtils.mkdir_p(tmp_dir)

  zip_path = File.join(tmp_dir, asset_name)

  puts "Downloading #{asset_name} for #{platform}..."
  download_file(download_url, zip_path)

  puts "Extracting #{asset_name}..."
  extract_dir = File.join(tmp_dir, platform)
  FileUtils.mkdir_p(extract_dir)
  extract_zip(zip_path, extract_dir)

  # Find all shared libraries with the correct extension
  lib_extension = config[:lib_extension]
  lib_files = Dir.glob(File.join(extract_dir, '**', "*#{lib_extension}"))

  # Filter out libraries we don't want to bundle
  excluded_libs = %w[libmtmd mtmd]
  lib_files.reject! do |lib_file|
    basename = File.basename(lib_file, lib_extension)
    excluded_libs.any? { |excluded| basename.include?(excluded) }
  end

  raise "Could not find any library files with extension #{lib_extension} in #{extract_dir}" if lib_files.empty?

  # Copy all libraries to lib directory
  lib_dest_dir = File.join('lib', 'rllama', platform)
  FileUtils.mkdir_p(lib_dest_dir)

  lib_files.each do |lib_file|
    dest_file = File.join(lib_dest_dir, File.basename(lib_file))
    FileUtils.cp(lib_file, dest_file)
    puts "Installed #{File.basename(lib_file)}"
  end

  puts "Successfully installed #{lib_files.length} libraries to #{lib_dest_dir}"

  # Clean up
  FileUtils.rm_f(zip_path)
  FileUtils.rm_rf(extract_dir)
end

def copy_local_build(platform, config)
  # Determine the correct local build directory based on platform
  local_build_dir = case platform
                    when 'aarch64-linux'
                      'build/linux-aarch64'
                    when 'aarch64-linux-musl'
                      'build/linux-aarch64-musl'
                    else
                      raise "Unsupported platform for local build: #{platform}"
                    end

  unless Dir.exist?(local_build_dir)
    build_task = platform == 'aarch64-linux-musl' ? 'llama:build_arm_musl' : 'llama:build_arm'
    raise "Local build directory not found: #{local_build_dir}\n" \
          "Please run 'rake #{build_task}' first to build the ARM64 binaries."
  end

  puts "Using local build from #{local_build_dir} for #{platform}..."

  # Find all shared libraries with the correct extension
  lib_extension = config[:lib_extension]
  lib_files = Dir.glob(File.join(local_build_dir, '**', "*#{lib_extension}"))

  # Filter out libraries we don't want to bundle
  excluded_libs = %w[libmtmd mtmd]
  lib_files.reject! do |lib_file|
    basename = File.basename(lib_file, lib_extension)
    excluded_libs.any? { |excluded| basename.include?(excluded) }
  end

  raise "Could not find any library files with extension #{lib_extension} in #{local_build_dir}" if lib_files.empty?

  # Copy all libraries to lib directory
  lib_dest_dir = File.join('lib', 'rllama', platform)
  FileUtils.mkdir_p(lib_dest_dir)

  lib_files.each do |lib_file|
    dest_file = File.join(lib_dest_dir, File.basename(lib_file))
    FileUtils.cp(lib_file, dest_file)
    puts "Installed #{File.basename(lib_file)}"
  end

  puts "Successfully installed #{lib_files.length} libraries from local build to #{lib_dest_dir}"
end

def download_file(url, destination)
  uri = URI(url)

  Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == 'https') do |http|
    request = Net::HTTP::Get.new(uri)

    http.request(request) do |response|
      case response
      when Net::HTTPRedirection
        location = response['location']
        puts "Following redirect to #{location}"
        download_file(location, destination)
      when Net::HTTPSuccess
        File.open(destination, 'wb') do |file|
          response.read_body do |chunk|
            file.write(chunk)
          end
        end
      else
        raise "Failed to download #{url}: #{response.code} #{response.message}"
      end
    end
  end
end

def extract_zip(zip_file, destination)
  Zip::File.open(zip_file) do |zip|
    zip.each do |entry|
      dest_path = File.join(destination, entry.name)
      FileUtils.mkdir_p(File.dirname(dest_path))
      entry.extract(dest_path) unless File.exist?(dest_path)
    end
  end
end

def build_platform_gem(platform)
  config = PLATFORMS[platform]
  gem_platform = config[:gem_platform]

  # Ensure libraries exist
  lib_dir = File.join('lib', 'rllama', platform)
  lib_extension = config[:lib_extension]
  lib_files = Dir.glob(File.join(lib_dir, "*#{lib_extension}"))

  if lib_files.empty?
    puts "No libraries found in #{lib_dir}, downloading..."
    download_llama_binary(platform)
  end

  # Build the gem with platform specification
  FileUtils.mkdir_p('pkg')
  sh "gem build rllama.gemspec --platform #{gem_platform}"

  # Move to pkg directory
  FileUtils.mv(Dir.glob('rllama-*.gem'), 'pkg/')
end

def parse_platforms(platforms_arg)
  if platforms_arg && !platforms_arg.empty?
    platforms_arg.split(',').map(&:strip)
  else
    PLATFORMS.keys
  end
end

def list_built_gems
  puts "\nBuilt gems:"
  gems = Dir.glob('pkg/*.gem')
  if gems.empty?
    puts '  No gems found in pkg/'
  else
    gems.each do |gem|
      size = File.size(gem) / 1024.0
      puts "  #{File.basename(gem)} (#{size.round(1)} KB)"
    end
  end
end

# Default task
task default: ['gem:build_base']
