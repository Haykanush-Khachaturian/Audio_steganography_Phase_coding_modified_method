#include <iostream>
#include <complex>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

std::complex<long double> i(0, 1);
const long double pi = 3.14159265358979323846;
const double eps = 0.0001;

class Phase_Coding_Method
{
public:
	Phase_Coding_Method() = default;
	Phase_Coding_Method(const Phase_Coding_Method&) = delete;
	Phase_Coding_Method& operator = (const Phase_Coding_Method&) = delete;

	static bool read_audio(const std::string& audio_file_name, std::vector<char>& audio)
	{
		std::ifstream input(audio_file_name, std::ios::in | std::ios::binary);
		if (!input.is_open())
			return false;
		std::cout << "Reading file ... ";
		audio.assign((std::istreambuf_iterator<char>(input)), (std::istreambuf_iterator<char>()));
		input.close();
		std::cout << "Done!\n";
		return true;
	}

	static long double Sin(long double x)
	{
		return 	std::abs(sin(x)) < eps ? 0 : sin(x);
	}

	static long double Cos(long double x)
	{
		return std::abs(cos(x)) < eps ? 0 : cos(x);
	}

	template <typename T>
	std::vector<std::complex<long double>> FFT_impl(std::vector<T>& A)
	{
		int N = A.size();
		if (N == 1)
			return { (std::complex<long double>)A[0] };
		std::complex<long double> WN = Cos(-2 * pi / N) + i * Sin(-2 * pi / N), W = 1;
		std::vector<std::complex<long double>> Aeven, Aodd, Yeven, Yodd, Y(N);
		for (int i = 0; i < N; ++i)
		{
			i % 2 == 0 ? Aeven.push_back(A[i]) : Aodd.push_back(A[i]);
		}
		Yeven = FFT_impl(Aeven);
		Yodd = FFT_impl(Aodd);
		for (int j = 0; j < N / 2; ++j)
		{
			Y[j] = Yeven[j] + W * Yodd[j];
			Y[j + N / 2] = Yeven[j] - W * Yodd[j];
			W = W * WN;
			std::complex<long double> temp(0, 0);
			if (abs(real(W)) < 0.00001)
			{
				W = i * (imag(W) + temp);
				W += 2;
				W -= 2;
			}
			else if (abs(imag(W)) < 0.00001)
			{
				W = real(W);
				W += i;
				W -= i;
			}
		}
		return Y;
	}
};

class Embedding final : public Phase_Coding_Method 
{
private:
	std::string m_message;
	std::vector<char> m_audio;
	std::pair<int, int> m_result; //{ key, message_size };
	int m_segments_count;
	int m_segment_size;
	int m_start;

private:
	void segmentation()
	{
		for (int j = 44; j < m_audio.size(); ++j)
		{
			if ((int)m_audio[j] != 0)
			{
				m_start = j;
				break;
			}
		}
		int pure_audio_size = m_audio.size() - m_start;
		int message_size = m_message.size() * 8;
		double temp = std::log2(2 * message_size);
		int key = static_cast<int>(ceil(temp) + 1);
		m_segment_size = std::pow(2, key);
		m_segments_count = 0;
		if (pure_audio_size % m_segment_size > 0)
		{
			m_segments_count = pure_audio_size / m_segment_size + 1;
			//m_audio.resize(m_audio.size() + (m_segments_count * m_segment_size - pure_audio_size), 0);
			pure_audio_size = m_segments_count * m_segment_size;
		}
		else
			m_segments_count = pure_audio_size / m_segment_size;
		m_result = { key, message_size };
	}

	void FFT(std::vector<std::vector<std::complex<long double>>>& sigma)
	{
		int j = 0;
		for (auto it = m_audio.begin() + m_start; j < m_segments_count; ++j)
		{
			if (it + m_segment_size > m_audio.end())
			{
				std::cout << "OOPS!!!!!" << std::endl;
				break;
			}
			std::vector<int> temp(it, it + m_segment_size);
			if (j == 0)
			{
				sigma.push_back(FFT_impl(temp));
				break; //
			}
			if (j != m_segments_count - 1)
				it += m_segment_size;
		}
	}

	static std::vector<char> string_to_bin(const std::string& message)
	{
		if (message.size() == 0)
			return {};
		std::vector<char> binaryMessage(8 * message.size(), 0);
		for (size_t k = 0; k < message.size(); ++k)
		{
			unsigned char temp = message[k];
			for (size_t j = 0; j < 8; ++j)
			{
				binaryMessage[8 * (k + 1) - (j + 1)] = '0' + (temp % 2);
				temp /= 2;
			}
		}
		return binaryMessage;
	}

	std::vector<char> iFFT_impl(std::vector<std::complex<long double>>& A)
	{
		std::vector<char> newAudio;
		std::reverse(next(A.begin()), A.end());
		std::vector<std::complex<long double>> vec;
		vec = FFT_impl(A);
		int sz = A.size();
		char ch;
		long double dNum = 0.0;
		int intNum = 0;
		for (auto& e : vec)
		{
			dNum = real(e) / sz;
			if (dNum - static_cast<int>(dNum) >= 0.444445)
			{
				intNum = static_cast<int>(dNum) + 1;
			}
			else if (dNum - static_cast<int>(dNum) <= -0.995)
			{
				intNum = static_cast<int>(dNum) - 1;
			}
			else
				intNum = static_cast<int>(dNum);
			ch = static_cast<char>(intNum);
			newAudio.push_back(ch);
		}
		return newAudio;
	}

	void iFFT(int new_size, const std::vector<std::vector<long double>>& magnitude, const std::vector<std::vector<long double>>& new_phase)
	{
		std::vector<char> newAudio;// (audio.begin(), audio.begin() + start);
		std::vector<std::vector<std::complex<long double>>> newSigma(m_segments_count);
		for (int j = 0; j < m_segments_count; ++j)
		{
			for (int k = 0; k < new_size; ++k)
			{
				newSigma[j].push_back(magnitude[j][k] * (Cos(new_phase[j][k] * pi / 180) + i * Sin(new_phase[j][k] * pi / 180)));
			}
			newAudio = iFFT_impl(newSigma[j]);
			for (int j = m_start; j < m_start + newAudio.size(); ++j)
				m_audio[j] = newAudio[j - m_start];
			break; //
		}
	}

	void save_new_audio_file()
	{
		std::string new_audio_file = "output.wav";
		std::ofstream output(new_audio_file, std::ios::binary);
		output.write((const char*)&m_audio[0], m_audio.size());
		output.close();
		std::cout << "\nThe stego-file was saved as 'output.wav'\n"
			"This is the embedding secret info:\t{ " << m_result.first << ", " << m_result.second << " }\n";
	}

public:
	Embedding(const std::string& message, const std::string& audio_file_name) 
		: m_message(message), m_audio(), m_result({ -1, -1 }), m_segments_count(0), m_segment_size(0), m_start(-1)
	{
		if (read_audio(audio_file_name, m_audio))
		{
			segmentation();
		}
	}

	std::pair<int, int> hide_message()
	{
		if (m_result == std::make_pair<int, int>(-1, -1))
		{
			std::cout << "Something went wrong ...\n";
			return m_result;
		}
		std::vector<std::vector<std::complex<long double>>> sigma;
		FFT(sigma);
		int new_size = sigma[0].size();
		std::vector<std::vector<long double>> magnitude(m_segments_count), phase(m_segments_count);
		for (size_t j = 0; j < sigma.size(); ++j)
		{
			for (size_t k = 0; k < sigma[j].size(); ++k)
			{
				magnitude[j].push_back(abs(sigma[j][k]));
				phase[j].push_back(arg(sigma[j][k]) * 180 / pi);
			}
		}
		/*std::vector<std::vector<double>> delta_phase(m_segments_count);
		delta_phase[0].insert(delta_phase[0].end(), new_size, 0);
		for (int j = 1; j < m_segments_count; ++j)
		{
			for (size_t k = 0; k < phase[j].size(); ++k)
			{
				delta_phase[j].push_back(phase[j][k] - phase[j - 1][k]);
			}
		}*/
		std::vector<std::vector<long double>> new_phase(m_segments_count);
		std::vector<char> binary_message = string_to_bin(m_message);
		int bin_message_size = binary_message.size();
		for (int j = 0; j < bin_message_size; ++j)
		{
			phase[0][m_segment_size / 2 - bin_message_size + j] = (binary_message[j] ==  '1') ? -90 : 90;
			phase[0][m_segment_size / 2 + bin_message_size - j] = -phase[0][m_segment_size / 2 - bin_message_size + j];
		}
		new_phase[0] = phase[0];
		/*for (int j = 1; j < m_segments_count; ++j)
		{
			newPhase[j] = phase[j];
			for (int k = 0; k < new_size; ++k)
			{
				new_phase[j].push_back(new _phase[j - 1][k] + delta_phase[j][k]);
			}
		}*/
		iFFT(new_size, magnitude, new_phase);
		save_new_audio_file();
		return m_result;
	}
};

class Extracting final : public Phase_Coding_Method
{
private:
	std::vector<char> m_audio;
	int m_message_size;
	int m_segment_size;
	int m_start;

private:
	char bin_to_symbol(const std::vector<char>& vec)
	{
		int num = 0;
		for (int i = 0; i < vec.size(); ++i)
		{
			if (vec[i] == '1')
			{
				num += pow(2, 7 - i);
			}
		}
		return num;
	}

	void segmentation()
	{
		for (int j = 44; j < m_audio.size(); ++j)
		{
			if (m_audio[j] != 0)
			{
				m_start = j;
				break;
			}
		}
	}

public:
	Extracting(const std::string& audio_file_name, const std::pair<int, int>& info) 
		: m_audio(), m_message_size(info.second), m_segment_size(std::pow(2, info.first)), m_start(-1)
	{
		if (read_audio(audio_file_name, m_audio))
		{
			segmentation();
		}
	}

	std::string find_message()
	{
		if (m_start == -1)
		{
			std::cout << "Something went wrong ...\n";
			return std::string();
		}
		std::vector<char> first_segment(m_audio.begin() + m_start, m_audio.begin() + m_start + m_segment_size);
		std::vector<std::complex<long double>> sigma = FFT_impl(first_segment);
		std::vector<long double> phase;
		for (int j = 0; j < m_segment_size; ++j)
		{
			phase.push_back(std::arg(sigma[j]) * 180 / pi);
		}
		std::vector<std::vector<char>> numbers(m_message_size);
		int tempK = 0;
		int k = 0;
		for (int j = phase.size() / 2 - m_message_size; j < phase.size() / 2; ++j)
		{
			if (tempK != 0 && tempK % 8 == 0)
			{
				++k;
			}
			if (phase[j] > 0)
				numbers[k].push_back('0');
			else
				numbers[k].push_back('1');

			tempK++;
		}
		std::string stego;
		for (int i = 0; i < numbers.size(); i++)
		{
			stego.push_back(bin_to_symbol(numbers[i]));
		}
		std::cout << "\nHidden message:\t" << stego << std::endl;
		return stego;
	}
};

int main()
{
	char choice;
	std::cout << "Press 1 or 2:\n1. Embedding\n2. Extracting\n";
	std::cin >> choice;
	std::string audio_file_name;
	switch (choice)
	{
		case '1':
		{
			std::string message;
			std::cout << "Enter the secret message\n";
			std::string dummy;
			std::getline(std::cin, dummy);
			std::getline(std::cin, message);
			std::cout << "Enter the audio file name\n";
			std::cin >> audio_file_name;
			audio_file_name += ".wav";
			Embedding embed(message, audio_file_name);
			embed.hide_message();
			break;
		}
		case '2':
		{
			std::cout << "Enter the audio file name\n";
			std::cin >> audio_file_name;
			audio_file_name += ".wav";
			std::cout << "Enter the embedding info\nKey = ";
			int key, message_size;
			std::cin >> key;
			std::cout << "Message size = ";
			std::cin >> message_size;
			if (key > 0 && message_size > 0)
			{
				std::pair<int, int> info = { key, message_size };
				Extracting extractor(audio_file_name, info);
				extractor.find_message();
			}
			else
				std::cout << "Not valid info\n";
		}
	}
}
