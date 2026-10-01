
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


# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="MacroSnap",
    page_icon="🥗",
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Current stable Flash model recommended by Google.
MODEL_NAME = "gemini-3.8-flash"

# Read secrets from Streamlit Cloud / .streamlit/secrets.toml
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

TWILIO_ACCOUNT_SID = st.secrets["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = st.secrets["TWILIO_AUTH_TOKEN"]
TWILIO_WHATSAPP_FROM = st.secrets["TWILIO_WHATSAPP_FROM"]
TWILIO_CONTENT_SID = st.secrets["TWILIO_CONTENT_SID"]


# ---------------------------------------------------------
# Clients
# ---------------------------------------------------------

@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)


@st.cache_resource
def get_twilio_client():
    return TwilioClient(
        TWILIO_ACCOUNT_SID,
        TWILIO_AUTH_TOKEN,
    )


gemini_client = get_gemini_client()
twilio_client = get_twilio_client()


# ---------------------------------------------------------
# UI helpers
# ---------------------------------------------------------

def render_message(message):
    with st.chat_message(message["role"]):

        if message["kind"] == "text":
            st.write(message["content"])

        elif message["kind"] == "image":
            st.image(message["content"])


def add_message(role, kind, content):
    st.session_state.messages.append(
        {
            "role": role,
            "kind": kind,
            "content": content,
        }
    )

    render_message(st.session_state.messages[-1])


# ---------------------------------------------------------
# Gemini
# ---------------------------------------------------------

def ask_gemini(parts):
    """
    Sends a request to Gemini.

    Retries temporary 503 / UNAVAILABLE errors
    before returning an error message.
    """

    max_attempts = 3

    for attempt in range(max_attempts):

        try:
            response = st.session_state.chat.send_message(parts)

            if response and response.text:
                return response.text

            return "Gemini returned an empty response."

        except Exception as error:

            error_text = str(error)

            # Temporary Gemini capacity/service problem
            if (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "high demand" in error_text.lower()
            ):

                if attempt < max_attempts - 1:
                    # Wait before trying again
                    time.sleep(3 * (attempt + 1))
                    continue

            return f"Sorry, something went wrong: {error_text}"

    return "Sorry, Gemini is temporarily unavailable. Please try again."


# ---------------------------------------------------------
# WhatsApp helpers
# ---------------------------------------------------------

def clean_whatsapp_text(text):

    if not text:
        return "No nutrition summary available."

    # Remove newlines and excessive spaces
    text = " ".join(text.split())

    # Keep WhatsApp message reasonably sized
    if len(text) > 1500:
        return text[:1500] + "..."

    return text


def send_whatsapp(to_number, user_name, summary):

    try:

        # Remove accidental whitespace
        to_number = to_number.strip()

        # Prevent users from entering "whatsapp:" twice
        if to_number.startswith("whatsapp:"):
            to_number = to_number.replace("whatsapp:", "", 1)

        # Ensure country-code format
        if not to_number.startswith("+"):
            return (
                False,
                "Please enter your WhatsApp number with country code, "
                "for example +919876543210.",
            )

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


# ---------------------------------------------------------
# Step 1: Onboarding
# ---------------------------------------------------------

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
                "Enter your number with country code. "
                "Example: +919876543210"
            ),
        )

        submitted = st.form_submit_button(
            "Let's go 🚀"
        )

    if submitted:

        if not name.strip() or not whatsapp_number.strip():

            st.warning(
                "Please fill in both your name and WhatsApp number."
            )

        elif not whatsapp_number.strip().startswith("+"):

            st.warning(
                "Please enter your WhatsApp number with country code, "
                "for example +919876543210."
            )

        else:

            st.session_state.name = name.strip()

            st.session_state.whatsapp_number = (
                whatsapp_number.strip()
            )

            # Create Gemini chat
            st.session_state.chat = gemini_client.chats.create(
                model=MODEL_NAME,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT
                ),
            )

            st.session_state.messages = []

            st.session_state.onboarded = True

            st.rerun()

    st.stop()


# ---------------------------------------------------------
# Step 2: Chat interface
# ---------------------------------------------------------

header_col, button_col = st.columns(
    [5, 2],
    vertical_alignment="center",
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

with header_col:

    st.title("🥗 MacroSnap")


# ---------------------------------------------------------
# WhatsApp button
# ---------------------------------------------------------

with button_col:

    # Enable the button once the user has interacted
    # with Gemini at least once.
    send_disabled = len(st.session_state.messages) < 3

    if st.button(
        "📤 Send to WhatsApp",
        disabled=send_disabled,
        use_container_width=True,
    ):

        with st.spinner(
            "Summarizing your day..."
        ):

            summary = ask_gemini(
                [SUMMARY_REQUEST_PROMPT]
            )

        # Don't send an error message to WhatsApp
        # if Gemini itself failed.
        if summary.startswith(
            "Sorry, something went wrong:"
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


# ---------------------------------------------------------
# User information
# ---------------------------------------------------------

st.caption(
    f"Logged in as {st.session_state.name} "
    f"- updates go to {st.session_state.whatsapp_number}"
)


# ---------------------------------------------------------
# Display previous messages
# ---------------------------------------------------------

if not st.session_state.messages:

    add_message(
        "assistant",
        "text",
        WELCOME_MESSAGE_TEMPLATE.format(
            name=st.session_state.name
        ),
    )

else:

    for message in st.session_state.messages:

        render_message(message)


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------

user_input = st.chat_input(
    "Ask a question, or attach a photo of your meal",
    accept_file=True,
    file_type=[
        "jpg",
        "jpeg",
        "png",
    ],
)


# ---------------------------------------------------------
# Process user input
# ---------------------------------------------------------

if user_input:

    photo = (
        user_input.files[0]
        if user_input.files
        else None
    )

    text = user_input.text

    parts = []


    # -----------------------------------------------------
    # Image input
    # -----------------------------------------------------

    if photo is not None:

        photo_bytes = photo.getvalue()

        add_message(
            "user",
            "image",
            photo_bytes,
        )

        parts.append(
            types.Part.from_bytes(
                data=photo_bytes,
                mime_type=photo.type,
            )
        )


    # -----------------------------------------------------
    # Text input
    # -----------------------------------------------------

    if text:

        add_message(
            "user",
            "text",
            text,
        )

        parts.append(text)

    elif photo is not None:

        parts.append(
            "What is this meal? "
            "Give me the estimated calories, "
            "protein, carbohydrates, and fat."
        )


    # -----------------------------------------------------
    # Gemini response
    # -----------------------------------------------------

    if parts:

        with st.spinner(
            "Crunching the numbers..."
        ):

            answer = ask_gemini(parts)

        add_message(
            "assistant",
            "text",
            answer,
        )
