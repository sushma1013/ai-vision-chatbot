import json
import time

import streamlit as st
from google import genai
from google.genai import types
from twilio.rest import Client as TwilioClient

from prompts import (
    SUMMARY_REQUEST_PROMPT,
    SYSTEM_PROMPT,
    WELCOME_MESSAGE_TEMPLATE,
)


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="MacroSnap",
    page_icon="🥗",
    layout="centered",
)


# ============================================================
# SECRETS / CONFIGURATION
# ============================================================

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

TWILIO_ACCOUNT_SID = st.secrets["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = st.secrets["TWILIO_AUTH_TOKEN"]
TWILIO_WHATSAPP_FROM = st.secrets["TWILIO_WHATSAPP_FROM"]
TWILIO_CONTENT_SID = st.secrets["TWILIO_CONTENT_SID"]


# ============================================================
# GEMINI MODEL
# ============================================================

MODEL_NAME = "gemini-2.5-flash"


# ============================================================
# CLIENT INITIALIZATION
# ============================================================

@st.cache_resource
def get_gemini_client():
    """
    Creates and caches the Gemini client.

    @st.cache_resource prevents Streamlit from creating
    a new Gemini client on every rerun.
    """
    return genai.Client(
        api_key=GEMINI_API_KEY
    )


@st.cache_resource
def get_twilio_client():
    """
    Creates and caches the Twilio client.
    """
    return TwilioClient(
        TWILIO_ACCOUNT_SID,
        TWILIO_AUTH_TOKEN,
    )


gemini_client = get_gemini_client()
twilio_client = get_twilio_client()


# ============================================================
# CREATE GEMINI CHAT
# ============================================================

def create_gemini_chat():
    """
    Creates a Gemini chat session with the application's
    system instructions.
    """

    return gemini_client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT
        ),
    )


# ============================================================
# MESSAGE RENDERING
# ============================================================

def render_message(message):
    """
    Displays a single message in the Streamlit chat UI.
    """

    with st.chat_message(message["role"]):

        if message["kind"] == "text":
            st.write(message["content"])

        elif message["kind"] == "image":
            st.image(
                message["content"],
                use_container_width=True,
            )


# ============================================================
# ADD MESSAGE
# ============================================================

def add_message(role, kind, content):
    """
    Adds a message to session state and immediately
    renders it.
    """

    st.session_state.messages.append(
        {
            "role": role,
            "kind": kind,
            "content": content,
        }
    )

    render_message(
        st.session_state.messages[-1]
    )


# ============================================================
# GEMINI REQUEST
# ============================================================

def ask_gemini(parts):
    """
    Sends a request to Gemini.

    Handles temporary 503 and 429 errors using
    exponential backoff.

    Retry sequence:

        Attempt 1
        ↓
        wait 2 seconds
        ↓
        Attempt 2
        ↓
        wait 4 seconds
        ↓
        Attempt 3
    """

    max_retries = 3

    for attempt in range(max_retries):

        try:

            response = st.session_state.chat.send_message(
                parts
            )

            if response is None:
                return "Gemini returned an empty response."

            if not response.text:
                return "Gemini returned an empty response."

            return response.text

        except Exception as error:

            error_text = str(error)

            # --------------------------------------------
            # Temporary Gemini server overload
            # --------------------------------------------

            is_503 = (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "high demand" in error_text.lower()
            )

            # --------------------------------------------
            # Rate limit / quota
            # --------------------------------------------

            is_429 = (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "quota" in error_text.lower()
            )

            # --------------------------------------------
            # Retry temporary errors
            # --------------------------------------------

            if is_503 or is_429:

                if attempt < max_retries - 1:

                    wait_time = 2 ** attempt

                    if is_503:
                        st.info(
                            f"Gemini is busy. "
                            f"Retrying in {wait_time} seconds..."
                        )

                    elif is_429:
                        st.info(
                            f"Gemini rate limit reached. "
                            f"Retrying in {wait_time} seconds..."
                        )

                    time.sleep(wait_time)

                    continue

                # ----------------------------------------
                # All retries failed
                # ----------------------------------------

                if is_503:

                    return (
                        "Gemini is currently experiencing "
                        "high demand. Please try again in "
                        "a few moments."
                    )

                if is_429:

                    return (
                        "Gemini API quota or rate limit "
                        "has been reached. Please check "
                        "your Gemini API usage and billing."
                    )

            # --------------------------------------------
            # Other errors
            # --------------------------------------------

            return (
                "Sorry, something went wrong while "
                f"processing your request.\n\n"
                f"Error: {error_text}"
            )

    return "Gemini is temporarily unavailable."


# ============================================================
# WHATSAPP TEXT CLEANING
# ============================================================

def clean_whatsapp_text(text):
    """
    Cleans Gemini's response before sending it to WhatsApp.

    Twilio content templates have message-size limitations,
    so we limit the summary to 1500 characters.
    """

    if not text:
        return "No nutrition summary available."

    # Collapse newlines and excessive whitespace
    text = " ".join(text.split())

    # Limit message length
    if len(text) > 1500:
        return text[:1500] + "..."

    return text


# ============================================================
# SEND WHATSAPP
# ============================================================

def send_whatsapp(to_number, user_name, summary):
    """
    Sends the daily nutrition summary through Twilio WhatsApp.
    """

    try:

        content_variables = json.dumps(
            {
                "1": user_name,
                "2": clean_whatsapp_text(summary),
            },
            ensure_ascii=False,
        )

        message = twilio_client.messages.create(
            from_=TWILIO_WHATSAPP_FROM,
            to=f"whatsapp:{to_number}",
            content_sid=TWILIO_CONTENT_SID,
            content_variables=content_variables,
        )

        return True, message.sid

    except Exception as error:

        return False, str(error)


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# STEP 1 — ONBOARDING
# ============================================================

if "onboarded" not in st.session_state:

    st.title("🥗 MacroSnap")

    st.caption(
        "Snap it. Track it. Text yourself the results."
    )

    with st.form("onboarding_form"):

        name = st.text_input(
            "Your name"
        )

        whatsapp_number = st.text_input(
            "WhatsApp number (with country code)",
            placeholder="+91XXXXXXXXXX",
            help=(
                "This is the number MacroSnap will "
                "text your summary to."
            ),
        )

        submitted = st.form_submit_button(
            "Let's go 🚀"
        )

    # ========================================================
    # FORM SUBMISSION
    # ========================================================

    if submitted:

        # --------------------------------------------
        # Validate input
        # --------------------------------------------

        if (
            not name.strip()
            or not whatsapp_number.strip()
        ):

            st.warning(
                "Please fill in both your name "
                "and WhatsApp number."
            )

        else:

            # ----------------------------------------
            # Save user information
            # ----------------------------------------

            st.session_state.name = name.strip()

            st.session_state.whatsapp_number = (
                whatsapp_number.strip()
            )

            # ----------------------------------------
            # Create Gemini conversation
            # ----------------------------------------

            try:

                st.session_state.chat = (
                    create_gemini_chat()
                )

            except Exception as error:

                st.error(
                    "Unable to initialize Gemini.\n\n"
                    f"{error}"
                )

                st.stop()

            # ----------------------------------------
            # Initialize chat
            # ----------------------------------------

            st.session_state.messages = []

            st.session_state.onboarded = True

            # ----------------------------------------
            # Rerun application
            # ----------------------------------------

            st.rerun()

    st.stop()


# ============================================================
# STEP 2 — MAIN CHAT INTERFACE
# ============================================================

header_col, button_col = st.columns(
    [5, 2],
    vertical_alignment="center",
)


# ============================================================
# HEADER
# ============================================================

with header_col:

    st.title("🥗 MacroSnap")


# ============================================================
# WHATSAPP BUTTON
# ============================================================

with button_col:

    # Require at least some conversation before
    # allowing daily summary.

    send_disabled = (
        len(st.session_state.messages) < 3
    )

    if st.button(
        "📤 Send to WhatsApp",
        disabled=send_disabled,
        use_container_width=True,
    ):

        with st.spinner(
            "Summarizing your day..."
        ):

            summary = ask_gemini(
                [
                    SUMMARY_REQUEST_PROMPT
                ]
            )

        # --------------------------------------------
        # Don't send an obvious Gemini error to WhatsApp
        # --------------------------------------------

        if (
            summary.startswith(
                "Gemini is currently experiencing"
            )
            or summary.startswith(
                "Gemini API quota"
            )
            or summary.startswith(
                "Sorry, something went wrong"
            )
        ):

            st.error(summary)

        else:

            success, info = send_whatsapp(
                st.session_state.whatsapp_number,
                st.session_state.name,
                summary,
            )

            if success:

                st.success(
                    "Sent! Check your WhatsApp 📲"
                )

            else:

                st.error(
                    f"Couldn't send that: {info}"
                )


# ============================================================
# USER INFORMATION
# ============================================================

st.caption(
    f"Logged in as "
    f"{st.session_state.name} "
    f"- updates go to "
    f"{st.session_state.whatsapp_number}"
)


# ============================================================
# WELCOME MESSAGE
# ============================================================

if not st.session_state.messages:

    add_message(
        "assistant",
        "text",
        WELCOME_MESSAGE_TEMPLATE.format(
            name=st.session_state.name
        ),
    )

else:

    # Render existing conversation
    for message in st.session_state.messages:

        render_message(message)


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask a question, or attach a photo of your meal",
    accept_file=True,
    file_type=[
        "jpg",
        "jpeg",
        "png",
    ],
)


# ============================================================
# PROCESS USER MESSAGE
# ============================================================

if user_input:

    photo = (
        user_input.files[0]
        if user_input.files
        else None
    )

    text = user_input.text

    parts = []


    # ========================================================
    # PROCESS IMAGE
    # ========================================================

    if photo is not None:

        photo_bytes = photo.getvalue()

        # --------------------------------------------
        # Display uploaded image
        # --------------------------------------------

        add_message(
            "user",
            "image",
            photo_bytes,
        )

        # --------------------------------------------
        # Convert image into Gemini Part
        # --------------------------------------------

        parts.append(
            types.Part.from_bytes(
                data=photo_bytes,
                mime_type=photo.type,
            )
        )


    # ========================================================
    # PROCESS TEXT
    # ========================================================

    if text:

        add_message(
            "user",
            "text",
            text,
        )

        parts.append(text)


    # ========================================================
    # IMAGE ONLY MESSAGE
    # ========================================================

    elif photo is not None:

        parts.append(
            "What is this meal? "
            "Identify the food and estimate "
            "the calories, protein, carbohydrates, "
            "and fat."
        )


    # ========================================================
    # SEND TO GEMINI
    # ========================================================

    if parts:

        with st.spinner(
            "🥗 Crunching the numbers..."
        ):

            answer = ask_gemini(parts)


        # ====================================================
        # DISPLAY GEMINI RESPONSE
        # ====================================================

        add_message(
            "assistant",
            "text",
            answer,
        )