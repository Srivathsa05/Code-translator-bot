import os
import re
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Set up logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

class CodeTranslator:
    def __init__(self):
        # Initialize with a pre-trained model for code translation
        # Note: You might need to experiment with different models
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/humangen-translate-code")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/humangen-translate-code")
        except Exception as e:
            logging.error(f"Model loading error: {e}")
            self.tokenizer = None
            self.model = None

    def translate_code(self, code, source_lang, target_lang):
        if not self.model or not self.tokenizer:
            return "Error: Translation model not loaded correctly."
        
        try:
            # Prepare the input for translation
            input_text = f"Translate from {source_lang} to {target_lang}: {code}"
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate translation
            outputs = self.model.generate(**inputs, max_length=512)
            translated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return translated_code
        except Exception as e:
            logging.error(f"Translation error: {e}")
            return f"Translation error: {str(e)}"

# Initialize the translator
translator = CodeTranslator()

# Telegram Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text(
        "👋 Welcome to Code Translator Bot! \n\n"
        "Send your code in this format:\n"
        "```python\n"
        "your code here\n"
        "``` to javascript\n\n"
        "Example:\n"
        "```python\n"
        "def greet(name):\n"
        "    print(f'Hello, {name}!')\n"
        "``` to javascript"
    )

async def translate_code(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Translate code when a message is sent."""
    user_text = update.message.text
    
    try:
        # Check if the message follows the expected format
        if "to " not in user_text.lower():
            await update.message.reply_text(
                "❌ Invalid format! Please use:\n"
                "```language\n"
                "your code\n"
                "``` to target_language"
            )
            return
        
        # Split the text into parts
        parts = user_text.split("to ")
        if len(parts) < 2:
            await update.message.reply_text("❌ Invalid code format. Specify target language.")
            return
        
        # Extract source language and code
        source_and_code = parts[0].strip()
        target_lang = parts[1].strip()
        
        # Check for code block
        code_match = re.search(r'```(\w+)\n(.*?)```', source_and_code, re.DOTALL)
        if not code_match:
            await update.message.reply_text("❌ Invalid code format. Use code blocks (```).")
            return
        
        source_lang = code_match.group(1)
        code = code_match.group(2).strip()
        
        # Translate the code
        translated_code = translator.translate_code(code, source_lang, target_lang)
        
        # Send the translated code
        response_message = (
            f"🔄 Translation from {source_lang} to {target_lang}:\n"
            f"```{target_lang}\n"
            f"{translated_code}\n"
            f"```"
        )
        await update.message.reply_text(response_message, parse_mode="Markdown")
    
    except Exception as e:
        logging.error(f"Error in translation: {e}")
        await update.message.reply_text(f"❌ An error occurred: {str(e)}")

def main():
    # Create the Application and pass it your bot's token
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    
    if not TELEGRAM_TOKEN:
        raise ValueError("No Telegram token found. Set TELEGRAM_TOKEN environment variable.")
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Register handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, translate_code))
    
    # Start the Bot
    logging.info("Bot is running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
