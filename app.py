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
# API SECRETS
# ============================================================

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

TWILIO_ACCOUNT_SID = st.secrets["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = st.secrets["TWILIO_AUTH_TOKEN"]
TWILIO_WHATSAPP_FROM = st.secrets["TWILIO_WHATSAPP_FROM"]
TWILIO_CONTENT_SID = st.secrets["TWILIO_CONTENT_SID"]


# ============================================================
# GEMINI MODELS
# ============================================================
#
# The first model is the primary model.
#
# If a model returns a 404/model unavailable error,
# the application automatically tries the next model.
#
# IMPORTANT:
# Model availability depends on your Google AI API project.
# Remove models that your account does not support.
#
# ============================================================

GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.8-flash-lite",
    "gemini-3.8-pro",
    "gemini-3.7-flash",
    "gemini-3.7-flash-lite",
]


# ============================================================
# RETRY CONFIGURATION
# ============================================================

MAX_RETRIES_PER_MODEL = 3

INITIAL_RETRY_DELAY = 2


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client():
    """
    Creates one Gemini client and reuses it
    across Streamlit reruns.
    """

    return genai.Client(
        api_key=GEMINI_API_KEY
    )


# ============================================================
# TWILIO CLIENT
# ============================================================

@st.cache_resource
def get_twilio_client():
    """
    Creates one Twilio client and reuses it.
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

def create_gemini_chat(model_name):
    """
    Creates a Gemini chat session for the selected model.
    """

    return gemini_client.chats.create(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT
        ),
    )


# ============================================================
# INITIALIZE GEMINI MODEL
# ============================================================

def initialize_gemini():

    if "current_model_index" not in st.session_state:
        st.session_state.current_model_index = 0

    if "current_model" not in st.session_state:
        st.session_state.current_model = GEMINI_MODELS[0]

    try:

        st.session_state.chat = create_gemini_chat(
            st.session_state.current_model
        )

        return True

    except Exception as error:

        st.session_state.gemini_error = str(error)

        return False


# ============================================================
# SWITCH TO NEXT MODEL
# ============================================================

def switch_to_next_model():
    """
    Switches to the next configured Gemini model.

    Returns:
        True  -> successfully created next chat
        False -> no models remaining
    """

    current_index = st.session_state.current_model_index

    next_index = current_index + 1

    if next_index >= len(GEMINI_MODELS):
        return False

    st.session_state.current_model_index = next_index

    next_model = GEMINI_MODELS[next_index]

    st.session_state.current_model = next_model

    try:

        st.session_state.chat = create_gemini_chat(
            next_model
        )

        return True

    except Exception as error:

        st.session_state.gemini_error = str(error)

        # Try another model recursively
        return switch_to_next_model()


# ============================================================
# ERROR CLASSIFICATION
# ============================================================

def is_model_unavailable(error_text):
    """
    Determines whether the error indicates that the
    current model cannot be used.
    """

    error_text = error_text.lower()

    return (
        "404" in error_text
        or "not_found" in error_text
        or "not found" in error_text
        or "no longer available" in error_text
        or "model is not available" in error_text
    )


def is_temporary_error(error_text):
    """
    Determines whether the error is temporary and
    should be retried.
    """

    error_text = error_text.lower()

    return (
        "503" in error_text
        or "unavailable" in error_text
        or "high demand" in error_text
        or "429" in error_text
        or "resource_exhausted" in error_text
        or "quota" in error_text
        or "rate limit" in error_text
    )


# ============================================================
# GEMINI REQUEST
# ============================================================

def ask_gemini(parts):
    """
    Sends a request to Gemini.

    Behavior:

    1. Use current model.
    2. Retry temporary 503/429 errors.
    3. If model is unavailable, switch to next model.
    4. Continue until models are exhausted.
    """

    models_checked = 0

    while (
        st.session_state.current_model_index
        < len(GEMINI_MODELS)
    ):

        current_model = st.session_state.current_model

        models_checked += 1

        # ====================================================
        # RETRY CURRENT MODEL
        # ====================================================

        for attempt in range(MAX_RETRIES_PER_MODEL):

            try:

                response = (
                    st.session_state.chat.send_message(
                        parts
                    )
                )

                if response is None:
                    return "Gemini returned an empty response."

                if not response.text:
                    return "Gemini returned an empty response."

                return response.text

            except Exception as error:

                error_text = str(error)

                # --------------------------------------------
                # MODEL UNAVAILABLE
                # --------------------------------------------

                if is_model_unavailable(error_text):

                    break


                # --------------------------------------------
                # TEMPORARY ERROR
                # --------------------------------------------

                if is_temporary_error(error_text):

                    if attempt < MAX_RETRIES_PER_MODEL - 1:

                        wait_time = (
                            INITIAL_RETRY_DELAY
                            * (2 ** attempt)
                        )

                        time.sleep(wait_time)

                        continue

                    # Current model exhausted retries
                    break


                # --------------------------------------------
                # UNKNOWN ERROR
                # --------------------------------------------

                return (
                    "Sorry, something went wrong while "
                    "processing your request.\n\n"
                    f"Error: {error_text}"
                )


        # ====================================================
        # SWITCH MODEL
        # ====================================================

        previous_model = current_model

        switched = switch_to_next_model()

        if switched:

            new_model = st.session_state.current_model

            st.warning(
                f"⚠️ Switching Gemini model: "
                f"{previous_model} → {new_model}"
            )

            continue

        # ====================================================
        # NO MODELS LEFT
        # ====================================================

        return (
            "❌ All configured Gemini models are "
            "currently unavailable.\n\n"
            "Please check your Gemini API model "
            "availability, quota, or billing."
        )

    return (
        "Gemini is currently unavailable. "
        "Please try again later."
    )


# ============================================================
# CLEAN WHATSAPP TEXT
# ============================================================

def clean_whatsapp_text(text):

    if not text:
        return "No nutrition summary available."

    # Remove excessive whitespace
    text = " ".join(text.split())

    # WhatsApp summary limit
    if len(text) > 1500:
        text = text[:1500] + "..."

    return text


# ============================================================
# SEND WHATSAPP
# ============================================================

# ============================================================
# WHATSAPP BUTTON
# ============================================================
has_user_messages = any(
    message["role"] == "user"
    for message in st.session_state.messages
)

with button_col:

    if st.button(
        "📤 Send to WhatsApp",
        disabled=not has_user_messages,
        use_container_width=True,
    ):

        with st.spinner(
            "📱 Creating your nutrition summary..."
        ):

            summary = ask_gemini(
                [SUMMARY_REQUEST_PROMPT]
            )

        # ----------------------------------------------------
        # Check whether Gemini returned an error
        # ----------------------------------------------------

        if (
            summary.startswith("❌")
            or summary.startswith("Gemini is")
            or summary.startswith("Sorry")
        ):

            st.error(summary)

        else:

            with st.spinner(
                "📲 Sending to WhatsApp..."
            ):

                success, info = send_whatsapp(
                    st.session_state.whatsapp_number,
                    st.session_state.name,
                    summary,
                )

            if success:

                st.success(
                    "✅ Sent successfully! "
                    "Check your WhatsApp 📲"
                )

                # Optional: display the Twilio message ID
                st.caption(
                    f"Message ID: {info}"
                )

            else:

                st.error(
                    f"❌ WhatsApp message failed:\n\n{info}"
                )


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


if "current_model_index" not in st.session_state:
    st.session_state.current_model_index = 0


if "current_model" not in st.session_state:
    st.session_state.current_model = GEMINI_MODELS[0]
# ============================================================
# MESSAGE RENDERING
# ============================================================

def render_message(message):
    """
    Render a single chat message.
    """

    with st.chat_message(message["role"]):

        if message["kind"] == "text":

            st.write(
                message["content"]
            )

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
    Add a message to Streamlit session state
    and render it immediately.
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
    # FORM SUBMITTED
    # ========================================================

    if submitted:

        if (
            not name.strip()
            or not whatsapp_number.strip()
        ):

            st.warning(
                "Please fill in both your name "
                "and WhatsApp number."
            )

        else:

            st.session_state.name = name.strip()

            st.session_state.whatsapp_number = (
                whatsapp_number.strip()
            )

            # --------------------------------------------
            # Reset model selection
            # --------------------------------------------

            st.session_state.current_model_index = 0

            st.session_state.current_model = (
                GEMINI_MODELS[0]
            )

            # --------------------------------------------
            # Create Gemini chat
            # --------------------------------------------

            if initialize_gemini():

                st.session_state.messages = []

                st.session_state.onboarded = True

                st.rerun()

            else:

                st.error(
                    "Unable to initialize Gemini."
                )

                st.error(
                    st.session_state.get(
                        "gemini_error",
                        "Unknown Gemini error.",
                    )
                )

    st.stop()


# ============================================================
# MAIN HEADER
# ============================================================

header_col, button_col = st.columns(
    [5, 2],
    vertical_alignment="center",
)


# ============================================================
# TITLE
# ============================================================

with header_col:

    st.title("🥗 MacroSnap")


# ============================================================
# CURRENT MODEL DISPLAY
# ============================================================

with header_col:

    st.caption(
        f"🤖 AI Model: "
        f"`{st.session_state.current_model}`"
    )


# ============================================================
# WHATSAPP BUTTON
# ============================================================

with button_col:

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


        # ====================================================
        # CHECK GEMINI ERROR
        # ====================================================

        is_error = (
            summary.startswith(
                "❌ All configured Gemini models"
            )
            or summary.startswith(
                "Gemini is currently unavailable"
            )
            or summary.startswith(
                "Sorry, something went wrong"
            )
            or summary.startswith(
                "Gemini API quota"
            )
        )


        if is_error:

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
# PROCESS USER INPUT
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
    # IMAGE
    # ========================================================

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


    # ========================================================
    # TEXT
    # ========================================================

    if text:

        add_message(
            "user",
            "text",
            text,
        )

        parts.append(text)


    # ========================================================
    # IMAGE WITHOUT TEXT
    # ========================================================

    elif photo is not None:

        parts.append(
            (
                "Analyze this meal carefully. "
                "Identify the food items and estimate "
                "the calories, protein, carbohydrates, "
                "and fat. Explain that the values are "
                "estimates."
            )
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
        # DISPLAY ANSWER
        # ====================================================

        add_message(
            "assistant",
            "text",
            answer,
        )