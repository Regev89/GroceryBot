import logging
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
# import telegram_LLM_to_list as sll
import Client.telegram_sql_handle as tsql
import requests
import os
import io
from google.cloud import speech, texttospeech
from pydub import AudioSegment
import sqlite3
import json

TOKEN = <Enter you TelegramBot API> 

# Ensure the GOOGLE_APPLICATION_CREDENTIALS environment variable is set
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = <Enter your JSON file from google cloud> #don't forget to add it to your repo as well

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Initialize the Google Cloud client (for audio)
client = speech.SpeechClient()

#####################
# service Functions #
#####################


### SQL ###
def init_db():
    conn = sqlite3.connect('grocerybot.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS groceries
                 (chat_id INTEGER PRIMARY KEY, list_data TEXT)''')
    conn.commit()
    conn.close()


init_db()


def get_json(user_input):  # request from server (kamatera)
    base_url = 'http://194.36.89.62:5001/analyze_nl_request'
    params = {
        'nl_request': user_input
    }
    try:
        response = requests.get(base_url, params=params)
        json_data = response.json()
        return json_data
    except Exception as e:
        print(f"An error occurred: {e}")


# using google's API text-to-speech
async def text_to_speech(text, lang='en-US', gender='NEUTRAL'):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=lang,
        ssml_gender=texttospeech.SsmlVoiceGender[gender]
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.OGG_OPUS
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    # Using BytesIO to avoid writing to disk
    audio_stream = io.BytesIO(response.audio_content)
    return audio_stream


async def handle_response(update, context, text):  # Returning both text and speech
    await update.message.reply_text(text)
    audio_stream = await text_to_speech(text)
    audio_stream.seek(0)  # Reset stream position
    await context.bot.send_voice(chat_id=update.effective_chat.id, voice=audio_stream)


#############################
# bot messages text / audio #
#############################

# using google's API speech-to-text
async def audio_message(update: Update, context):
    file = await context.bot.get_file(update.message.voice.file_id)
    voice_data = await file.download_as_bytearray()

    # Convert from bytearray to audio segment directly in memory
    sound = AudioSegment.from_file(io.BytesIO(voice_data), format="ogg")
    sound = sound.set_frame_rate(
        16000).set_sample_width(2)  # 2 bytes per sample

    # Export directly to BytesIO object
    wav_stream = io.BytesIO()
    sound.export(wav_stream, format="wav")
    wav_stream.seek(0)

    # Load WAV directly from BytesIO for processing
    client = speech.SpeechClient()
    content = wav_stream.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US"
    )

    # Attempt to recognize speech
    try:
        from_google = client.recognize(config=config, audio=audio)
        transcription = ' '.join(
            [result.alternatives[0].transcript for result in from_google.results])

        if not transcription.strip():
            await update.message.reply_text("I couldn't understand that audio. Please try again or send text.")
        else:
            print(transcription)
            if transcription.lower() == 'show list':
                await show_list(update, context)
            elif transcription.lower() == 'empty list':
                await empty_list(update, context)
            else:
                response = get_json(transcription)
                if response['valid']:
                    chat_id = update.effective_chat.id
                    conn = sqlite3.connect('grocerybot.db')
                    c = conn.cursor()
                    c.execute("SELECT list_data FROM groceries WHERE chat_id = ?",
                              (chat_id,))  # (chat_id,) replace the '?'
                    result = c.fetchone()
                    if result is not None:
                        retrieved_json = result[0]
                        retrieved_list = json.loads(retrieved_json)
                    else:
                        retrieved_list = []
                    conn.close()
                    llm_answer = tsql.answer_to_list(
                        response['groceries'], retrieved_list, chat_id)
                    await handle_response(update, context, llm_answer)
                else:
                    error_message = "I'm sorry but I am a grocery list agent. I can only respond to shopping related sentences."
                    if 'error_code' in response:
                        if response['error_code'] == 100:
                            error_message = "I'm sorry, I didn't fully understand your request. Please repeat."
                        elif response['error_code'] == 200:
                            error_message = "I'm sorry, I didn't fully understand your request. Please repeat."
                    await handle_response(update, context, error_message)

    except Exception as e:
        await update.message.reply_text(f"Error processing the audio: {e}")


# any text message
async def bot_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Processes user messages and replies based on grocery list operations."""
    user_input = update.message.text
    response = get_json(user_input)

    if response['valid']:
        chat_id = update.effective_chat.id
        conn = sqlite3.connect('grocerybot.db')
        c = conn.cursor()
        c.execute("SELECT list_data FROM groceries WHERE chat_id = ?",
                  (chat_id,))  # (chat_id,) replace the '?'
        result = c.fetchone()
        if result is not None:
            retrieved_json = result[0]
            retrieved_list = json.loads(retrieved_json)
        else:
            retrieved_list = []
        conn.close()
        llm_answer = tsql.answer_to_list(
            response['groceries'], retrieved_list, chat_id)
        await handle_response(update, context, llm_answer)
    else:
        error_message = "I'm sorry, I can only respond to shopping related sentences."
        if 'error_code' in response:
            if response['error_code'] == 100:
                error_message = "I'm sorry, I didn't fully understand your request. Please repeat."
            elif response['error_code'] == 200:
                error_message = "I'm sorry, I didn't fully understand your request. Please repeat."
        await handle_response(update, context, error_message)

################
# bot commands #
################


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!. My Name is GroceryBot. I can help you make an updating shopping list! Please tell me what groceries do you want.",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""

    await update.message.reply_text("""Here are the commands you can use:\n /list - show grocery list\n
/empty - empty grocery list""")


async def show_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Showing the grocery list """
    chat_id = update.effective_chat.id
    conn = sqlite3.connect('grocerybot.db')
    c = conn.cursor()
    c.execute("SELECT list_data FROM groceries WHERE chat_id = ?",
              (chat_id,))  # (chat_id,) replace the '?'
    result = c.fetchone()
    if result is not None:
        retrieved_json = result[0]
        retrieved_list = json.loads(retrieved_json)
    else:
        retrieved_list = []
    conn.close()

    if len(retrieved_list) > 0:
        response_text = f'Here is your list so far:\n{tsql.print_list(retrieved_list)}'
        await update.message.reply_text(response_text)
    else:
        response_text = "Your list is still empty."
        await handle_response(update, context, response_text)


async def empty_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Emptying the grocery list """
    chat_id = update.effective_chat.id
    conn = sqlite3.connect('grocerybot.db')
    c = conn.cursor()
    c.execute("DELETE FROM groceries WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()

    response_text = "List is now empty."
    await handle_response(update, context, response_text)


async def print_db(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    conn = sqlite3.connect('grocerybot.db')
    cursor = conn.cursor()

    # Query to select all records from the groceries table
    query = "SELECT * FROM groceries"
    cursor.execute(query)

    # Fetch all rows from the query result
    rows = cursor.fetchall()

    print('\nDatabase:\n')
    # Check if rows are fetched
    if rows:
        for row in rows:
            print(f'{row}\n')
    else:
        print("No data found in the 'groceries' table.")

    # Close the connection
    conn.close()


def main() -> None:
    """Start the bot."""
    application = Application.builder().token(TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("list", show_list))
    application.add_handler(CommandHandler("empty", empty_list))
    application.add_handler(CommandHandler("print", print_db))

    # handle non-command messages
    application.add_handler(MessageHandler(filters.VOICE, audio_message))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, bot_reply))

    application.run_polling()


if __name__ == "__main__":
    main()
