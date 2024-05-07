# Deep Learning Project with SLM and Groq LLM

## Description
This repository contains a deep learning project utilizing the MixBread MXBAI model and Groq LLM via LangChain. The project integrates various APIs including Groq, OpenAI, TelegramBot, and Google Cloud to provide a comprehensive application. The goal of this project is to [Briefly describe the main goal or application of your project].

## Demo 
<a href="https://drive.google.com/file/d/1aREqAK6W2xvVaB_lMn5l8TuLKrXdzpQz/view?usp=sharing" target="_blank">Link to video!</a>

## Features
- Utilization of the state-of-the-art SLM model (MixBread MXBAI).
- Integration with Groq's LLM through LangChain.
- Communication and control through a Telegram bot.
- Cloud-based operations and storage using Google Cloud.

## Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.8+
- Groq API key
- OpenAI API key
- TelegramBot API key
- Google Cloud credentials

## Installation
To install the necessary libraries, run the following command:
`pip install -r requirements.txt`

## Setup
1. Clone the repository:
`git clone https://github.com/Regev89/GroceryBot/`
`cd GroceryBot`

2. Create a `.env` file in the root directory and populate it with your API keys:
- GROQ_API_KEY='your_groq_api_key'
- OPENAI_API_KEY='your_openai_api_key'  

(You may use one of the two, no acctual need for both, but code should be edited accordingly.)
- TELEGRAM_BOT_API_KEY='your_telegram_bot_api_key'  -> MUST create with Telegram's BotFather
- GOOGLE_CLOUD_CREDENTIALS='path_to_your_google_cloud_credentials.json' -> for Text-to-Speech and Speech-to-Text fitures

## Usage
To run the project, execute:
`python main.py`
Ensure you have configured the `.env` file with your API keys as the project requires these to interact with the respective services.

## Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Your Name - [regev89@gmail.com](mailto:regev89@gmail.com)

## Acknowledgments
- mixedbread-ai/mxbai-embed-large-v1
- Groq LLM and LangChain
- OpenAI
- TelegramBot API
- Google Cloud

## References
- Sean Lee, Aamir Shakir, Darius Koenig, Julius Lipp. (2024). "Open Source Strikes Bread - New Fluffy Embeddings Model." Retrieved from [MixedBread.ai](https://www.mixedbread.ai/blog/mxbai-embed-large-v1).
- Xianming Li, Jing Li. (2023). "AnglE-optimized Text Embeddings." arXiv preprint arXiv:2309.12871. Available at [arXiv.org](https://arxiv.org/abs/2309.12871).


