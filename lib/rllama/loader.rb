# frozen_string_literal: true

require 'uri'
require 'net/http'
require 'fileutils'

module Rllama
  module Loader
    HUGGINGFACE_BASE_URL = 'https://huggingface.co'
    DEFAULT_DIR = File.join(Dir.home, '.rllama')

    UNITS = %w[B KB MB GB TB].freeze

    module_function

    def resolve(path_or_name, dir: nil)
      dir ||= DEFAULT_DIR

      dir = File.join(dir, 'models')

      return path_or_name if local_file?(path_or_name)

      if url?(path_or_name)
        download_from_url(path_or_name, dir)
      elsif huggingface_path?(path_or_name)
        download_from_huggingface(path_or_name, dir)
      else
        raise Error, "Invalid model path or name: #{path_or_name}"
      end
    end

    def local_file?(path)
      File.exist?(path)
    end

    def url?(path)
      uri = URI.parse(path)

      uri.is_a?(URI::HTTP) || uri.is_a?(URI::HTTPS)
    rescue URI::InvalidURIError
      false
    end

    def huggingface_path?(path)
      return false if path.start_with?('/') || path.include?('://')

      parts = path.split('/')

      parts.length >= 3 && parts.last.end_with?('.gguf')
    end

    def download_from_huggingface(hf_path, dir)
      parts = hf_path.split('/')

      raise Error, "Invalid HuggingFace path: #{hf_path}" if parts.length < 3

      org = parts[0]
      repo = parts[1]
      file_path = parts[2..].join('/')

      url = "#{HUGGINGFACE_BASE_URL}/#{org}/#{repo}/resolve/main/#{file_path}"

      local_path = File.join(dir, org, repo, file_path)

      return local_path if File.exist?(local_path)

      puts "Destination: #{local_path}"

      download_file(url, local_path, "HuggingFace model: #{hf_path}")
    end

    def download_from_url(url, dir)
      uri = URI.parse(url)

      filename = File.basename(uri.path)

      local_path = File.join(dir, filename)

      return local_path if File.exist?(local_path)

      puts "Destination: #{local_path}"

      download_file(url, local_path, "URL: #{url}")
    end

    def download_file(url, local_path, description)
      FileUtils.mkdir_p(File.dirname(local_path))

      temp_path = File.join(File.dirname(local_path), "~#{File.basename(local_path)}")

      existing_size = File.exist?(temp_path) ? File.size(temp_path) : 0

      uri = URI.parse(url)

      Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == 'https') do |http|
        request = Net::HTTP::Get.new(uri.request_uri)

        request['Range'] = "bytes=#{existing_size}-" if existing_size.positive?

        http.request(request) do |response|
          case response
          when Net::HTTPSuccess, Net::HTTPPartialContent
            if response['Content-Range']
              total_size = response['Content-Range'].split('/').last.to_i
            else
              total_size = response['content-length'].to_i

              if existing_size.positive? && response.code == '200'
                puts "\nServer doesn't support resume, starting from beginning..."

                existing_size = 0

                FileUtils.rm_f(temp_path)
              end
            end

            downloaded = existing_size
            file_mode = existing_size.positive? ? 'ab' : 'wb'

            File.open(temp_path, file_mode) do |file|
              response.read_body do |chunk|
                file.write(chunk)
                downloaded += chunk.size

                if total_size.positive?
                  progress = (downloaded.to_f / total_size * 100).round
                  total_str = format_bytes(total_size)
                  downloaded_str = format_bytes(downloaded)
                  padding = total_str.length
                  formatted_downloaded = format("%#{padding}s", downloaded_str)
                  print format("\rProgress: %<progress>6d%% (%<downloaded>s / %<total>s)",
                               progress: progress, downloaded: formatted_downloaded, total: total_str)
                else
                  print "\rDownloaded: #{format_bytes(downloaded)}"
                end
              end
            end

            unless verify_download(temp_path, total_size)
              FileUtils.rm_f(temp_path)

              raise Error, 'Download verification failed - file size mismatch'
            end

            File.rename(temp_path, local_path)

            puts
          when Net::HTTPRedirection
            redirect_url = response['location']

            redirect_url = URI.join(url, redirect_url).to_s unless redirect_url.start_with?('http://', 'https://')

            return download_file(redirect_url, local_path, description)
          when Net::HTTPRequestedRangeNotSatisfiable
            if File.exist?(temp_path)
              uri = URI.parse(url)

              Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == 'https') do |check_http|
                head_request = Net::HTTP::Head.new(uri.request_uri)
                head_response = check_http.request(head_request)

                if head_response.is_a?(Net::HTTPSuccess)
                  expected_size = head_response['content-length'].to_i
                  actual_size = File.size(temp_path)

                  if expected_size.positive? && expected_size == actual_size
                    File.rename(temp_path, local_path)

                    return local_path
                  end
                end
              end

              File.delete(temp_path)

              return download_file(url, local_path, description)
            end

            raise Error, "Range request failed: #{response.code} #{response.message}"
          else
            raise Error, "Failed to download model: #{response.code} #{response.message}"
          end
        end
      end

      local_path
    end

    def verify_download(local_path, expected_size)
      return true if expected_size <= 0

      actual_size = File.size(local_path)
      actual_size == expected_size
    end

    def format_bytes(bytes)
      return '0 B' if bytes.zero?

      exp = (Math.log(bytes) / Math.log(1024)).floor

      exp = [exp, UNITS.length - 1].min

      value = bytes.to_f / (1024**exp)

      if exp >= 3
        format('%<val>.2f %<unit>s', val: value, unit: UNITS[exp])
      else
        format('%<val>d %<unit>s', val: value.round, unit: UNITS[exp])
      end
    end
  end
end
