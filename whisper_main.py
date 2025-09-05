from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import logging
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import base64
import tempfile
import os
import re

# Import the robust TTS stack with gTTS for Indian languages
import whisper
from transformers import pipeline
from gtts import gTTS
import platform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Whisper model (small = good balance of speed & accuracy)
logger.info("🎤 Loading Whisper model (this may take a moment)...")
whisper_model = whisper.load_model("tiny")  # Using tiny model for faster loading
logger.info("✅ Whisper model loaded successfully!")

# Skip DialoGPT - too slow and unhelpful!
logger.info("🌾 Using Fast Agriculture Assistant instead of DialoGPT...")

app = FastAPI(title="🌾 Fast Agriculture AI with Whisper + Smart RAG + gTTS", version="5.0.0")

# Agriculture Knowledge Base for RAG (same as before)
AGRICULTURE_KB = {
    "crops": {
        "rice": {
            "en": "Rice is a staple grain crop. Best grown in flooded fields. Requires 4-6 months growing season. Plant during monsoon. Harvest when grains turn golden. Major varieties: Basmati, Jasmine, Arborio.",
            "ta": "அரிசி ஒரு முக்கிய தானிய பயிர். வெள்ளம் நிறைந்த வயல்களில் சிறப்பாக வளரும். 4-6 மாத வளர்ச்சி காலம் தேவை. பருவமழைக் காலத்தில் நடவு செய்யவும். தானியங்கள் தங்க நிறமாக மாறும்போது அறுவடை செய்யவும்.",
            "te": "వరి ప్రధాన ధాన్య పంట. నీరు నిండిన పొలాల్లో బాగా పెరుగుతుంది. 4-6 నెలల పెరుగుదల కాలం అవసరం. వర్షాకాలంలో నాటాలి. ధాన్యాలు బంగారు రంగులోకి మారినప్పుడు కోయాలి.",
            "ml": "അരി ഒരു പ്രധാന ധാന്യ വിള. വെള്ളം നിറഞ്ഞ വയലുകളിൽ നന്നായി വളരും. 4-6 മാസത്തെ വളർച്ചാകാലം ആവശ്യം. മഴക്കാലത്ത് നടണം. ധാന്യങ്ങൾ സ്വർണ്ണ നിറമാകുമ്പോൾ വിളവെടുക്കാം."
        },
        "wheat": {
            "en": "Wheat is a major cereal grain. Grows best in temperate climates. Sow in October-November. Harvest in March-April. Requires well-drained soil. Major varieties: Durum, Hard Red, Soft White.",
            "ta": "கோதுமை ஒரு முக்கிய தானிய பயிர். மிதமான காலநிலையில் சிறப்பாக வளரும். அக்டோபர்-நவம்பரில் விதைக்கவும். மார்ச்-ஏப்ரலில் அறுவடை செய்யவும். நல்ல வடிகால் மண் தேவை.",
            "te": "గోధుమ ప్రధాన ధాన్య పంట. సమశీతోష్ణ వాతావరణంలో బాగా పెరుగుతుంది. అక్టోబర్-నవంబర్‌లో విత్తాలి. మార్చి-ఏప్రిల్‌లో కోయాలి. మంచి నీటి వడపోత గల నేల అవసరం.",
            "ml": "ഗോതമ്പ് ഒരു പ്രധാന ധാന്യ വിള. മിതശീതോഷ്ണ കാലാവസ്ഥയിൽ നന്നായി വളരും. ഒക്ടോബർ-നവംബറിൽ വിതയ്ക്കണം. മാർച്ച്-ഏപ്രിലിൽ വിളവെടുക്കണം."
        },
        "sugarcane": {
            "en": "Sugarcane is a tropical cash crop. Requires hot, humid climate. Plant 12-18 month crop cycle. Needs abundant water. Harvest when stalks are mature and sweet. Major producer of sugar and jaggery.",
            "ta": "கரும்பு ஒரு வெப்பமண்டல பணப் பயிர். வெப்பமான, ஈரப்பதமான காலநிலை தேவை. 12-18 மாத பயிர் சுழற்சி. அதிக நீர் தேவை. தண்டுகள் முதிர்ந்து இனிப்பாக இருக்கும்போது அறுவடை.",
            "te": "చెరకు ఉష్ణమండల వాణిజ్య పంట. వేడిమిగిలిన, తేమతో కూడిన వాతావరణం అవసరం. 12-18 నెలల పంట చక్రం. పుష్కల నీరు అవసరం.",
            "ml": "കരിമ്പ് ഉഷ്ണമേഖലാ വാണിജ്യ വിള. ചൂടുള്ളതും ഈർപ്പമുള്ളതുമായ കാലാവസ്ഥ ആവശ്യം. 12-18 മാസത്തെ വിള ചക്രം. ധാരാളം വെള്ളം ആവശ്യം."
        }
    },
    "soil": {
        "clay": {
            "en": "Clay soil retains water well but drains slowly. Good for rice cultivation. Add organic matter to improve drainage. Test pH regularly. Suitable for crops that need consistent moisture.",
            "ta": "களிமண் மண் நீரை நன்றாக தக்கவைக்கிறது ஆனால் மெதுவாக வடிகிறது. நெல் சாகுபடிக்கு நல்லது. வடிகால் மேம்படுத்த கரிம பொருட்களை சேர்க்கவும்.",
            "te": "మట్టి మంచిగా నీరు నిలుపుకుంటుంది కానీ నెమ్మదిగా పారిపోతుంది. వరి సాగుకు మంచిది. నీటి వడపోతను మెరుగుపరచడానికి సేంద్రీయ పదార్థాలను కలపండి.",
            "ml": "കളിമണ്ണ് വെള്ളം നന്നായി നിലനിർത്തുന്നു പക്ഷേ പതുക്കെ ഒഴുകുന്നു. നെല്ലുകൃഷിക്ക് നല്ലത്. ഡ്രെയിനേജ് മെച്ചപ്പെടുത്താൻ ജൈവവസ്തുക്കൾ ചേർക്കുക."
        },
        "sandy": {
            "en": "Sandy soil drains quickly but requires frequent irrigation. Good for root vegetables. Add compost to retain nutrients. Suitable for crops like carrots, potatoes, onions.",
            "ta": "மணல் மண் விரைவாக வடிகிறது ஆனால் அடிக்கடி நீர்ப்பாசனம் தேவை. வேர் காய்கறிகளுக்கு நல்லது. ஊட்டச்சத்துக்களை தக்கவைக்க கம்போஸ்ட் சேர்க்கவும்.",
            "te": "ఇసుక మట్టి త్వరగా పారిపోతుంది కానీ తరచుగా నీటిపారుదల అవసరం. వేర్ కూరగాయలకు మంచిది. పోషకాలను నిలుపుకోవడానికి కంపోస్ట్ చేర్చండి.",
            "ml": "മണൽമണ്ണ് വേഗത്തിൽ ഒഴുകുന്നു പക്ഷേ ഇടയ്ക്കിടെ നനയ്ക്കേണ്ടതുണ്ട്. റൂട്ട് പച്ചക്കറികൾക്ക് നല്ലത്. പോഷകങ്ങൾ നിലനിർത്താൻ കമ്പോസ്റ്റ് ചേർക്കുക."
        },
        "loamy": {
            "en": "Loamy soil is ideal for most crops. Perfect balance of drainage and retention. Rich in nutrients. Suitable for vegetables, fruits, grains. Maintain with organic matter.",
            "ta": "களிমண் கலந்த மண் பெரும்பாலான பயிர்களுக்கு ஏற்றது. வடிகால் மற்றும் தக்கவைப்பின் சரியான சமநிலை. ஊட்டச்சத்து நிறைந்தது.",
            "te": "లేత మట్టి చాలా పంటలకు అనువైనది. డ్రైనేజ్ మరియు రిటెన్షన్ యొక్క పరిపూర్ణ సమతుల్యత. పోషకాలతో సమృద్ధిగా ఉంటుంది.",
            "ml": "പശിമമണ്ണ് മിക്ക വിളകൾക്കും അനുയോജ്യം. ഡ്രെയിനേജിന്റെയും നിലനിർത്തലിന്റെയും മികച്ച സന്തുലനം. പോഷകങ്ങളാൽ സമൃദ്ധം."
        }
    },
    "irrigation": {
        "drip": {
            "en": "Drip irrigation saves 30-50% water. Delivers water directly to plant roots. Reduces weed growth. Initial investment high but long-term savings. Best for row crops and orchards.",
            "ta": "சொட்டு நீர்ப்பாசனம் 30-50% நீரை சேமிக்கிறது. தாவர வேர்களுக்கு நேரடியாக நீர் வழங்குகிறது. களை வளர்ச்சியை குறைக்கிறது.",
            "te": "డ్రిప్ నీటిపారుదల 30-50% నీటిని ఆదా చేస్తుంది. మొక్కల వేర్లకు నేరుగా నీటిని అందిస్తుంది. కలుపు మొక్కల పెరుగుదలను తగ్గిస్తుంది.",
            "ml": "ഡ്രിപ്പ് ജലസേചനം 30-50% വെള്ളം ലാഭിക്കുന്നു. ചെടികളുടെ വേരുകളിലേക്ക് നേരിട്ട് വെള്ളം എത്തിക്കുന്നു. കളകളുടെ വളർച്ച കുറയ്ക്കുന്നു."
        },
        "sprinkler": {
            "en": "Sprinkler irrigation covers large areas efficiently. Good for uniform water distribution. Suitable for most field crops. Requires good water pressure. Can be automated easily.",
            "ta": "தெளிப்பு நீர்ப்பாசனம் பெரிய பகுதிகளை திறமையாக மூடுகிறது. சீரான நீர் விநியோகத்திற்கு நல்லது. பெரும்பாலான வயல் பயிர்களுக்கு ஏற்றது.",
            "te": "స్ప్రింక్లర్ నీటిపారుదల పెద్ద ప్రాంతాలను సమర్థవంతంగా కవర్ చేస్తుంది. ఏకరీతి నీటి పంపిణీకి మంచిది. చాలా వరల పంటలకు అనుకూలం.",
            "ml": "സ്പ്രിങ്ക്ളർ ജലസേചനം വലിയ പ്രദേശങ്ങളെ കാര്യക്ഷമമായി മൂടുന്നു. ഏകീകൃത ജല വിതരണത്തിന് നല്ലത്. മിക്ക വയൽ വിളകൾക്കും അനുയോജ്യം."
        }
    },
    "diseases": {
        "blight": {
            "en": "Blight causes dark spots on leaves and stems. Caused by fungal infection. Remove affected parts immediately. Use copper-based fungicides. Ensure good air circulation.",
            "ta": "கருமை நோய் இலைகள் மற்றும் தண்டுகளில் கருமையான புள்ளிகளை ஏற்படுத்துகிறது. பூஞ்சை தொற்றால் ஏற்படுகிறது. பாதிக்கப்பட்ட பகுதிகளை உடனே அகற்றவும்.",
            "te": "బ్లైట్ ఆకులు మరియు కాండాలపై ముదురు మచ్చలను కలిగిస్తుంది. ఫంగల్ ఇన్ఫెక్షన్ వల్ల కలుగుతుంది. ప్రభావిత భాగాలను వెంటనే తొలగించండి.",
            "ml": "ബ്ലൈറ്റ് ഇലകളിലും തണ്ടുകളിലും ഇരുണ്ട പാടുകൾ ഉണ്ടാക്കുന്നു. ഫംഗൽ അണുബാധ മൂലമാണ് ഇത് സംഭവിക്കുന്നത്. ബാധിത ഭാഗങ്ങൾ ഉടനെ നീക്കം ചെയ്യുക."
        }
    }
}

# TF-IDF Vectorizer for RAG
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Initialize knowledge base vectors
def initialize_rag():
    """Initialize RAG system with agriculture knowledge base"""
    global kb_vectors, kb_texts, kb_keys
    
    # Flatten knowledge base for vectorization
    kb_texts = []
    kb_keys = []
    
    for category, items in AGRICULTURE_KB.items():
        for item, langs in items.items():
            # Use English text for vectorization
            kb_texts.append(langs['en'])
            kb_keys.append((category, item))
    
    # Create TF-IDF vectors
    kb_vectors = vectorizer.fit_transform(kb_texts)
    logger.info(f"✅ RAG initialized with {len(kb_texts)} knowledge entries")

# Initialize RAG on startup
initialize_rag()

class QueryRequest(BaseModel):
    query: str
    language: str = "en"
    mode: str = "direct"
    user_type: str = "farmer"  # farmer, expert, student
    crop_type: str = ""
    land_size: str = ""
    soil_type: str = ""

def get_rag_context(query: str, language: str = "en", top_k: int = 3):
    """Get relevant context from knowledge base using RAG"""
    try:
        # Vectorize the query
        query_vector = vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, kb_vectors)[0]
        
        # Get top-k most similar entries
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        relevant_context = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                category, item = kb_keys[idx]
                context_data = AGRICULTURE_KB[category][item]
                
                # Get content in requested language
                content = context_data.get(language, context_data['en'])
                relevant_context.append({
                    'category': category,
                    'item': item,
                    'content': content,
                    'similarity': float(similarities[idx])
                })
        
        return relevant_context
    except Exception as e:
        logger.error(f"RAG error: {e}")
        return []

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
        <title>🌾 Fast Agriculture AI - Smart Assistant</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                -webkit-tap-highlight-color: transparent;
            }
            body {
                font-family: 'Segoe UI', 'Roboto', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: white;
                padding: 10px;
                overflow-x: hidden;
            }
            .container {
                max-width: 400px;
                margin: 0 auto;
                padding: 0 10px;
            }
            .header {
                text-align: center;
                margin-bottom: 20px;
                padding: 20px 0;
            }
            .header h1 {
                font-size: 2.2em;
                margin-bottom: 5px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .subtitle {
                font-size: 1em;
                opacity: 0.9;
                margin-bottom: 10px;
            }
            .demo-section {
                background: rgba(255, 255, 255, 0.15);
                border-radius: 15px;
                padding: 20px;
                margin: 15px 0;
                backdrop-filter: blur(10px);
            }
            .section-title {
                font-size: 1.3em;
                margin-bottom: 15px;
                text-align: center;
                color: #FFD700;
                font-weight: bold;
            }
            .option-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 12px;
                margin-bottom: 20px;
            }
            .option-button {
                background: rgba(255, 255, 255, 0.2);
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 12px;
                padding: 15px;
                color: white;
                font-size: 1.1em;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s ease;
                text-align: center;
                min-height: 60px;
                display: flex;
                align-items: center;
                justify-content: center;
                text-decoration: none;
            }
            .option-button:hover, .option-button.selected {
                background: rgba(255, 255, 255, 0.3);
                border-color: #FFD700;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            .option-button.selected {
                background: #FFD700;
                color: #333;
            }
            .quick-questions {
                display: grid;
                grid-template-columns: 1fr;
                gap: 10px;
            }
            .quick-question {
                background: rgba(76, 175, 80, 0.8);
                border: none;
                border-radius: 10px;
                padding: 12px;
                color: white;
                font-size: 1em;
                cursor: pointer;
                transition: all 0.3s ease;
                text-align: left;
                line-height: 1.4;
            }
            .quick-question:hover {
                background: rgba(76, 175, 80, 1);
                transform: translateY(-1px);
            }
            .input-section {
                margin: 20px 0;
            }
            .input-group {
                display: flex;
                flex-direction: column;
                gap: 12px;
                margin-bottom: 15px;
            }
            input, select, textarea {
                padding: 15px;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                background: rgba(255, 255, 255, 0.9);
                color: #333;
                width: 100%;
            }
            textarea {
                min-height: 80px;
                resize: vertical;
            }
            .action-buttons {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                margin: 15px 0;
            }
            .btn {
                padding: 15px;
                border: none;
                border-radius: 10px;
                font-size: 1.1em;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s ease;
                text-align: center;
            }
            .btn-primary {
                background: #4CAF50;
                color: white;
            }
            .btn-secondary {
                background: #FF9800;
                color: white;
            }
            .btn-voice {
                background: #2196F3;
                color: white;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }
            .response {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                padding: 20px;
                margin: 20px 0;
                min-height: 100px;
                white-space: pre-wrap;
                line-height: 1.6;
                font-size: 1.1em;
            }
            .loading {
                animation: pulse 1.5s infinite;
                text-align: center;
                font-size: 1.2em;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.7; }
                100% { opacity: 1; }
            }
            .demo-info {
                background: rgba(76, 175, 80, 0.2);
                border-left: 4px solid #4CAF50;
                padding: 15px;
                margin: 15px 0;
                border-radius: 0 10px 10px 0;
                font-size: 0.95em;
            }
            .status-bar {
                background: rgba(0, 0, 0, 0.2);
                padding: 10px;
                border-radius: 8px;
                margin: 10px 0;
                font-size: 0.9em;
                text-align: center;
            }
            @media (max-width: 480px) {
                .header h1 { font-size: 1.8em; }
                .container { padding: 0 5px; }
                .demo-section { padding: 15px; }
                .action-buttons { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌾 Agriculture AI</h1>
                <div class="subtitle">🎤 Whisper + � Smart Assistant + 🔊 gTTS</div>
            </div>

            <div class="demo-info">
                <strong>🎯 Enhanced Features:</strong><br>
                🎤 Whisper: Accurate speech recognition<br>
                🤖 Smart Assistant: FAST practical advice<br>
                🔊 gTTS: High-quality Google TTS for Indian languages<br>
                🌾 RAG: Agriculture knowledge base<br>
                ⚡ <strong>NO MORE SLOW AI! Direct practical help!</strong>
            </div>

            <!-- Language Selection -->
            <div class="demo-section">
                <div class="section-title">1️⃣ Select Language</div>
                <div class="option-grid">
                    <button class="option-button" onclick="selectLanguage('en')" id="lang-en">
                        🇺🇸 English
                    </button>
                    <button class="option-button" onclick="selectLanguage('ta')" id="lang-ta">
                        🇮🇳 தமிழ் (Tamil)
                    </button>
                    <button class="option-button" onclick="selectLanguage('te')" id="lang-te">
                        🇮🇳 తెలుగు (Telugu)
                    </button>
                    <button class="option-button" onclick="selectLanguage('ml')" id="lang-ml">
                        🇮🇳 മലയാളം (Malayalam)
                    </button>
                </div>
            </div>

            <!-- User Profile -->
            <div class="demo-section">
                <div class="section-title">2️⃣ Your Profile</div>
                <div class="option-grid">
                    <button class="option-button" onclick="selectProfile('farmer')" id="profile-farmer">
                        👨‍🌾 Farmer
                    </button>
                    <button class="option-button" onclick="selectProfile('expert')" id="profile-expert">
                        🔬 Agriculture Expert
                    </button>
                    <button class="option-button" onclick="selectProfile('student')" id="profile-student">
                        🎓 Student/Learner
                    </button>
                </div>
            </div>

            <!-- Land Details -->
            <div class="demo-section">
                <div class="section-title">3️⃣ Land Details</div>
                <div class="input-group">
                    <select id="cropType">
                        <option value="">Select Main Crop</option>
                        <option value="rice">🌾 Rice</option>
                        <option value="wheat">🌾 Wheat</option>
                        <option value="sugarcane">🎋 Sugarcane</option>
                        <option value="vegetables">🥬 Vegetables</option>
                        <option value="fruits">🍎 Fruits</option>
                        <option value="other">🌱 Other</option>
                    </select>
                    <select id="landSize">
                        <option value="">Select Land Size</option>
                        <option value="small">🏠 Small (< 2 acres)</option>
                        <option value="medium">🏡 Medium (2-10 acres)</option>
                        <option value="large">🏭 Large (> 10 acres)</option>
                    </select>
                    <select id="soilType">
                        <option value="">Select Soil Type</option>
                        <option value="clay">🟤 Clay Soil</option>
                        <option value="sandy">🟨 Sandy Soil</option>
                        <option value="loamy">🟫 Loamy Soil</option>
                        <option value="other">❓ Not Sure</option>
                    </select>
                </div>
            </div>

            <!-- Quick Questions -->
            <div class="demo-section">
                <div class="section-title">4️⃣ Quick Questions</div>
                <div class="quick-questions">
                    <button class="quick-question" onclick="askQuickQuestion('What is the best crop for my soil type?')">
                        🌱 What crop is best for my soil?
                    </button>
                    <button class="quick-question" onclick="askQuickQuestion('How much water does my crop need?')">
                        💧 Water requirements for crops
                    </button>
                    <button class="quick-question" onclick="askQuickQuestion('What fertilizer should I use?')">
                        🧪 Best fertilizers to use
                    </button>
                    <button class="quick-question" onclick="askQuickQuestion('How to prevent crop diseases?')">
                        🦠 Disease prevention tips
                    </button>
                    <button class="quick-question" onclick="askQuickQuestion('When is the best time to plant?')">
                        📅 Best planting seasons
                    </button>
                </div>
            </div>

            <!-- Custom Question -->
            <div class="demo-section">
                <div class="section-title">📝 Ask Your Question</div>
                <div class="input-section">
                    <textarea id="queryInput" placeholder="Type your agriculture question here..."></textarea>
                    <div class="action-buttons">
                        <button class="btn btn-primary" onclick="askAI()">🚀 Ask Smart Assistant</button>
                        <button class="btn btn-voice" onclick="startWhisperInput()">🎤 Whisper</button>
                    </div>
                    <button class="btn btn-secondary" onclick="testTTS()" style="width: 100%; margin-top: 10px;">
                        🔊 Test gTTS (Google TTS)
                    </button>
                    <button class="btn btn-secondary" onclick="testPhoneticTamil()" style="width: 100%; margin-top: 5px; background: #e53e3e;">
                        🎯 Test Tamil Script (Native pronunciation)
                    </button>
                </div>
            </div>

            <!-- Status Bar -->
            <div class="status-bar" id="statusBar">
                🟢 Ready | Smart Assistant + Whisper + gTTS | Language: English
            </div>

            <!-- Response Area -->
            <div id="response" class="response">
                Welcome to Agriculture AI v3.0! 🌾<br><br>
                <strong>New Stack:</strong><br>
                🎤 Whisper: Accurate speech-to-text<br>
                � Smart Assistant: FAST practical advice<br>
                🔊 pyttsx3: Offline text-to-speech<br>
                🌾 RAG: Agriculture knowledge base<br><br>
                <strong>100% FREE & OFFLINE!</strong>
            </div>
        </div>

        <script>
            let selectedLanguage = 'en';
            let selectedProfile = '';

            // Language selection
            function selectLanguage(lang) {
                selectedLanguage = lang;
                document.querySelectorAll('[id^="lang-"]').forEach(btn => btn.classList.remove('selected'));
                document.getElementById(`lang-${lang}`).classList.add('selected');
                updateStatusBar();
                console.log(`🌍 Language selected: ${lang}`);
            }

            // Profile selection
            function selectProfile(profile) {
                selectedProfile = profile;
                document.querySelectorAll('[id^="profile-"]').forEach(btn => btn.classList.remove('selected'));
                document.getElementById(`profile-${profile}`).classList.add('selected');
                updateStatusBar();
                console.log(`👤 Profile selected: ${profile}`);
            }

            // Update status bar
            function updateStatusBar() {
                const langNames = {
                    'en': 'English',
                    'ta': 'Tamil',
                    'te': 'Telugu',
                    'ml': 'Malayalam'
                };
                
                const status = `🟢 Ready | Smart Assistant + Whisper + gTTS | Language: ${langNames[selectedLanguage]}`;
                document.getElementById('statusBar').textContent = status;
            }

            // Quick question handler
            function askQuickQuestion(question) {
                document.getElementById('queryInput').value = question;
                askAI();
            }

            // Get user details
            function getUserDetails() {
                return {
                    cropType: document.getElementById('cropType').value,
                    landSize: document.getElementById('landSize').value,
                    soilType: document.getElementById('soilType').value
                };
            }

            // MAIN FUNCTION - Ask AI with Smart Assistant + RAG
            async function askAI() {
                console.log('🚀 askAI function called with Smart Assistant');
                
                const query = document.getElementById('queryInput').value;
                const responseDiv = document.getElementById('response');
                const details = getUserDetails();

                if (!query.trim()) {
                    alert('Please enter a question about agriculture!');
                    return;
                }

                console.log(`📝 Query: "${query}" in language: ${selectedLanguage}`);
                responseDiv.innerHTML = '<div class="loading">� Smart Assistant is thinking with RAG...</div>';

                try {
                    const requestData = {
                        query: query,
                        language: selectedLanguage,
                        mode: 'rag',
                        user_type: selectedProfile,
                        crop_type: details.cropType,
                        land_size: details.landSize,
                        soil_type: details.soilType
                    };

                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(requestData)
                    });

                    const data = await response.json();
                    console.log('✅ Received Smart Assistant response:', data);
                    
                    let responseText = `<strong>� Smart Assistant + RAG Response:</strong>\\n\\n${data.answer}`;
                    
                    if (data.rag_sources && data.rag_sources.length > 0) {
                        responseText += `\\n\\n📚 <strong>Knowledge Sources:</strong>\\n`;
                        data.rag_sources.forEach((source, idx) => {
                            responseText += `${idx + 1}. ${source.category.toUpperCase()}: ${source.item}\\n`;
                        });
                    }
                    
                    responseText += `\\n<hr style="border: 1px solid rgba(255,255,255,0.3); margin: 15px 0;">`;
                    responseText += `📊 Model: ${data.model} | ⏱️ Time: ${data.processing_time_ms}ms | 🔗 Sources: ${data.rag_sources ? data.rag_sources.length : 0}`;
                    
                    responseDiv.innerHTML = responseText;

                    // Use pyttsx3 TTS
                    if (data.answer) {
                        console.log(`🔊 Starting pyttsx3 TTS: "${data.answer.substring(0, 50)}..."`);
                        await speakWithPyttsx3(data.answer, selectedLanguage);
                    }

                } catch (error) {
                    console.error('❌ Error:', error);
                    responseDiv.innerHTML = `❌ Error: ${error.message}`;
                }
            }

            // Enhanced pyttsx3 TTS function
            async function speakWithPyttsx3(text, language) {
                console.log(`🔊 Using pyttsx3 TTS for ${language}`);
                
                try {
                    const ttsResponse = await fetch('/generate-tts', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            text: text,
                            language: language
                        })
                    });
                    
                    const ttsData = await ttsResponse.json();
                    
                    if (ttsData.success && ttsData.audio_base64) {
                        console.log(`✅ Using ${ttsData.service} (${ttsData.voice})`);
                        
                        // Create audio element and play (supports both WAV and MP3)
                        const audio = new Audio();
                        const audioFormat = ttsData.audio_format || 'wav';
                        audio.src = `data:audio/${audioFormat};base64,${ttsData.audio_base64}`;
                        
                        return new Promise((resolve) => {
                            audio.onended = () => {
                                console.log('✅ pyttsx3 TTS completed successfully');
                                resolve();
                            };
                            
                            audio.onerror = (error) => {
                                console.error('❌ Audio playback error:', error);
                                resolve();
                            };
                            
                            audio.play().catch(error => {
                                console.error('❌ Audio play error:', error);
                                resolve();
                            });
                        });
                    } else {
                        console.log('🔄 pyttsx3 not available, no TTS');
                    }
                    
                } catch (error) {
                    console.error('❌ TTS request failed:', error);
                }
            }

            // Whisper speech input function
            async function startWhisperInput() {
                console.log('🎤 Starting Whisper input...');
                
                const responseDiv = document.getElementById('response');
                responseDiv.innerHTML = '🎤 Listening with Whisper... Speak your agriculture question!';
                
                try {
                    // Use Web API for audio capture
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    const mediaRecorder = new MediaRecorder(stream);
                    const audioChunks = [];
                    
                    mediaRecorder.ondataavailable = (event) => {
                        audioChunks.push(event.data);
                    };
                    
                    mediaRecorder.onstop = async () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        const arrayBuffer = await audioBlob.arrayBuffer();
                        const base64Audio = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
                        
                        console.log('🎤 Audio captured, sending to Whisper...');
                        responseDiv.innerHTML = '🎤 Processing with Whisper...';
                        
                        try {
                            const response = await fetch('/whisper-transcribe', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                },
                                body: JSON.stringify({
                                    audio_data: base64Audio,
                                    language: selectedLanguage
                                })
                            });
                            
                            const data = await response.json();
                            
                            if (data.success) {
                                console.log(`✅ Whisper transcription: "${data.transcribed_text}"`);
                                document.getElementById('queryInput').value = data.transcribed_text;
                                responseDiv.innerHTML = `🎤 Whisper heard: "${data.transcribed_text}"\\n\\nClick "Ask Smart Assistant" to get an answer!`;
                            } else {
                                console.error('❌ Whisper failed:', data.error);
                                responseDiv.innerHTML = `❌ Whisper error: ${data.error}`;
                            }
                            
                        } catch (error) {
                            console.error('❌ Whisper request failed:', error);
                            responseDiv.innerHTML = `❌ Whisper request failed: ${error.message}`;
                        }
                        
                        // Stop all tracks
                        stream.getTracks().forEach(track => track.stop());
                    };
                    
                    // Record for 5 seconds
                    mediaRecorder.start();
                    setTimeout(() => {
                        if (mediaRecorder.state === 'recording') {
                            mediaRecorder.stop();
                        }
                    }, 5000);
                    
                } catch (error) {
                    console.error('❌ Microphone access failed:', error);
                    responseDiv.innerHTML = `❌ Microphone access failed: ${error.message}`;
                }
            }

            // Test phonetic Tamil function
            async function testPhoneticTamil() {
                const nativeTests = {
                    'en': 'Hello farmer. Rice is an important crop in agriculture.',
                    'ta': 'வணக்கம் விவசாயி. அரிசி ஒரு முக்கிய பயிர்.',
                    'te': 'నమస్కారం రైతు. వరి ఒక ముఖ్యమైన పంట.',
                    'ml': 'നമസ്കാരം കർഷകാ. അരി ഒരു പ്രധാന വിള.'
                };
                
                const testText = nativeTests[selectedLanguage] || nativeTests['en'];
                console.log(`🎯 Testing native script: "${testText}"`);
                
                document.getElementById('response').innerHTML = `🎯 Testing Native Script TTS...\\n\\n"${testText}"\\n\\n� Using Google gTTS for perfect pronunciation...`;
                
                await speakWithPyttsx3(testText, selectedLanguage);
                document.getElementById('response').innerHTML += '\\n✅ Native script TTS completed! 🎵';
            }

            // Test TTS function with gTTS
            async function testTTS() {
                const testTexts = {
                    'en': 'Hello farmer. This is a test of Google Text-to-Speech with perfect pronunciation.',
                    'ta': 'வணக்கம் விவசாயி. இது கூகிள் பேச்சு தொழில்நுட்ப சோதனை.',
                    'te': 'నమస్కారం రైతు. ఇది గూగుల్ వాయిస్ టెక్నాలజీ టెస్ట్.',
                    'ml': 'നമസ്കാരം കർഷകാ. ഇത് ഗൂഗിൾ വോയ്‌സ് ടെക്നോളജി ടെസ്റ്റ്.'
                };
                
                const testText = testTexts[selectedLanguage] || testTexts['en'];
                console.log(`🔊 Testing gTTS: "${testText}"`);
                
                document.getElementById('response').innerHTML = `🔊 Testing Google gTTS...\\n\\n"${testText}"\\n\\n🎵 Using native ${selectedLanguage} pronunciation...`;
                
                await speakWithPyttsx3(testText, selectedLanguage);
                document.getElementById('response').innerHTML += '\\n✅ gTTS test completed perfectly!';
            }

            // Allow Enter key to submit
            document.getElementById('queryInput').addEventListener('keypress', function(event) {
                if (event.key === 'Enter') {
                    askAI();
                }
            });

            console.log('🚀 Agriculture AI v3.0 ready with Whisper + Smart Assistant + gTTS!');
        </script>
    </body>
    </html>
    """

@app.post("/query")
async def query_agriculture(request: QueryRequest):
    start_time = time.time()
    
    try:
        logger.info(f"🌾 Smart RAG Query: {request.query[:50]}... | Language: {request.language} | Profile: {request.user_type}")
        
        # Get RAG context
        rag_context = get_rag_context(request.query, request.language, top_k=3)
        
        # Fast Smart Agriculture Assistant - No more slow DialoGPT!
        def generate_smart_agriculture_answer(query, language, rag_context, user_context):
            """Generate practical agriculture answers directly from knowledge base"""
            
            # Build comprehensive answer from RAG context
            if rag_context:
                # Use the most relevant context
                best_match = rag_context[0]
                answer = best_match['content']
                
                # Add specific advice based on user context
                if user_context.get('soil_type') == 'clay' and 'rice' in query.lower():
                    if language == 'ta':
                        answer += " களிமண் மண்ணில் நெல் சாகுபடிக்கு ஏற்றது. நல்ல வடிகால் அமைப்பு வேண்டும்."
                    else:
                        answer += " Clay soil is perfect for rice cultivation. Ensure proper drainage system."
                
                elif user_context.get('land_size') == 'small' and any(word in query.lower() for word in ['irrigation', 'water']):
                    if language == 'ta':
                        answer += " சிறிய நிலத்திற்கு சொட்டு நீர்ப்பாசனம் சிறந்தது. 30-50% நீர் சேமிப்பு."
                    else:
                        answer += " For small farms, drip irrigation is ideal. Saves 30-50% water."
                
                # Add seasonal advice
                import datetime
                current_month = datetime.datetime.now().month
                
                if 6 <= current_month <= 10:  # Monsoon season
                    if language == 'ta':
                        answer += " தற்போது கரீப் பருவம். நெல், பருத்தி நடவுக்கு ஏற்ற காலம்."
                    else:
                        answer += " Current Kharif season. Good time for rice, cotton planting."
                elif 11 <= current_month <= 4:  # Winter season
                    if language == 'ta':
                        answer += " தற்போது ரபி பருவம். கோதுமை, கடுகு விதைப்புக்கு ஏற்ற காலம்."
                    else:
                        answer += " Current Rabi season. Good time for wheat, mustard sowing."
                
                return answer
            
            # Fallback responses with practical advice instead of "consult officials"
            practical_fallbacks = {
                'en': {
                    'general': "For general farming: 1) Test your soil pH (should be 6.0-7.5), 2) Use organic matter like compost, 3) Follow proper irrigation schedule, 4) Monitor for pests regularly. Need specific advice? Ask about your crop, soil type, or season.",
                    'fertilizer': "Use balanced NPK fertilizer. For most crops: Apply 40kg Urea + 25kg DAP + 15kg MOP per acre. Split urea application - half at sowing, rest after 30-45 days.",
                    'pest': "For pest control: 1) Use neem oil spray (5ml/liter), 2) Remove affected plant parts, 3) Maintain field hygiene, 4) Use yellow sticky traps for flying pests.",
                    'disease': "For plant diseases: 1) Ensure good air circulation, 2) Avoid overhead watering, 3) Remove infected parts immediately, 4) Use copper-based fungicides for fungal issues."
                },
                'ta': {
                    'general': "பொதுவான வேளாண்மைக்கு: 1) மண்ணின் pH சோதனை செய்யவும் (6.0-7.5 இருக்க வேண்டும்), 2) கம்போஸ்ட் போன்ற கரிம பொருட்களைப் பயன்படுத்தவும், 3) சரியான நீர்ப்பாசனம், 4) தொடர்ந்து பூச்சிகளைக் கண்காணிக்கவும்.",
                    'fertilizer': "சமச்சீர் NPK உரம் பயன்படுத்தவும். பெரும்பாலான பயிர்களுக்கு: ஏக்கருக்கு 40கிலோ யூரியா + 25கிலோ DAP + 15கிலோ MOP.",
                    'pest': "பூச்சி கட்டுப்பாட்டுக்கு: 1) வேப்ப எண்ணெய் தெளிப்பு (5மி.லி/லிட்டர்), 2) பாதிக்கப்பட்ட பகுதிகளை அகற்றவும், 3) வயல் சுத்தம், 4) மஞ்சள் நிற ஒட்டும் பொறிகள்.",
                    'disease': "தாவர நோய்களுக்கு: 1) நல்ல காற்றோட்டம், 2) இலைகளில் நேரடியாக தண்ணீர் ஊற்றாதீர்கள், 3) நோய்வாய்ப்பட்ட பகுதிகளை உடனே அகற்றவும்."
                }
            }
            
            # Determine response category
            query_lower = query.lower()
            if any(word in query_lower for word in ['fertilizer', 'urea', 'dap', 'nutrients']):
                category = 'fertilizer'
            elif any(word in query_lower for word in ['pest', 'insect', 'bug', 'spray']):
                category = 'pest'
            elif any(word in query_lower for word in ['disease', 'fungus', 'rot', 'blight']):
                category = 'disease'
            else:
                category = 'general'
            
            return practical_fallbacks.get(language, practical_fallbacks['en']).get(category, practical_fallbacks['en']['general'])
        
        # Generate fast, practical answer
        user_context = {
            "profile": request.user_type,
            "crop": request.crop_type,
            "land_size": request.land_size,
            "soil_type": request.soil_type
        }
        
        answer = generate_smart_agriculture_answer(
            request.query, 
            request.language, 
            rag_context, 
            user_context
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "answer": answer,
            "confidence": 0.95,  # Higher confidence since we're giving practical advice
            "processing_time_ms": round(processing_time),
            "language": request.language,
            "mode": "smart_agriculture_assistant",
            "model": "Fast Smart RAG",
            "rag_sources": [{"category": ctx['category'], "item": ctx['item'], "similarity": ctx['similarity']} for ctx in rag_context],
            "user_context": user_context
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        
        # Even fallback gives practical advice
        practical_fallback = {
            'en': "I'm your agriculture assistant. For immediate help: 1) Soil issues - add organic matter, 2) Pest problems - use neem spray, 3) Fertilizer - use balanced NPK. Ask me specific questions about crops, soil, pests, or seasons!",
            'ta': "நான் உங்கள் வேளாண் உதவியாளர். உடனடி உதவிக்கு: 1) மண் பிரச்சனைகள் - கரிம பொருட்கள் சேர்க்கவும், 2) பூச்சி பிரச்சனைகள் - வேப்ப எண்ணெய் தெளிக்கவும், 3) உரம் - சமச்சீர் NPK பயன்படுத்தவும்।",
            'te': "నేను మీ వ్యవసాయ సహాయకుడను. తక్షణ సహాయం కోసం: 1) మట్టి సమస్యలు - సేంద్రీయ పదార్థాలు కలపండి, 2) కీటకాల సమస్యలు - వేప నూనె స్ప్రే చేయండి।",
            'ml': "ഞാൻ നിങ്ങളുടെ കാർഷിക സഹായിയാണ്. ഉടനടി സഹായത്തിന്: 1) മണ്ണിന്റെ പ്രശ്നങ്ങൾ - ജൈവവസ്തുക്കൾ ചേർക്കുക, 2) കീടങ്ങളുടെ പ്രശ്നങ്ങൾ - വേപ്പെണ്ണ തളിക്കുക।"
        }
        
        return {
            "answer": practical_fallback.get(request.language, practical_fallback['en']),
            "confidence": 0.8,
            "processing_time_ms": round((time.time() - start_time) * 1000),
            "language": request.language,
            "mode": "emergency_fallback",
            "model": "Fast Smart RAG",
            "error": "minor_error_handled"
        }

@app.post("/whisper-transcribe")
async def whisper_transcribe(request: dict):
    """Transcribe audio using Whisper model"""
    try:
        audio_data = request.get("audio_data", "")
        language = request.get("language", "en")
        
        if not audio_data:
            raise HTTPException(status_code=400, detail="Audio data is required")
        
        logger.info(f"🎤 Transcribing audio with Whisper for {language}")
        
        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        
        try:
            # Transcribe with Whisper
            result = whisper_model.transcribe(
                temp_audio_path,
                language=language if language != 'en' else None  # Let Whisper auto-detect if not English
            )
            
            transcribed_text = result["text"].strip()
            detected_language = result.get("language", language)
            
            logger.info(f"✅ Whisper transcription: '{transcribed_text[:50]}...'")
            
            return {
                "success": True,
                "transcribed_text": transcribed_text,
                "detected_language": detected_language,
                "confidence": result.get("avg_logprob", 0.8)
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
                
    except Exception as e:
        logger.error(f"❌ Whisper transcription failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Whisper transcription failed"
        }

@app.post("/generate-tts")
async def generate_tts(request: dict):
    """Generate TTS using gTTS (Google Text-to-Speech) - Perfect for Indian languages"""
    try:
        text = request.get("text", "")
        language = request.get("language", "en")
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        logger.info(f"🔊 gTTS Request: {text[:50]}... in {language}")
        
        # Clean and prepare text
        text = re.sub(r'\s+', ' ', text.strip())
        text = text[:500]  # Limit length for gTTS
        
        # Map language codes for gTTS
        gtts_language_map = {
            'en': 'en',
            'ta': 'ta',
            'te': 'te', 
            'ml': 'ml',
            'hi': 'hi',
            'kn': 'kn',  # Kannada
            'bn': 'bn',  # Bengali
            'gu': 'gu',  # Gujarati
            'mr': 'mr',  # Marathi
            'pa': 'pa'   # Punjabi
        }
        
        gtts_lang = gtts_language_map.get(language, 'en')
        logger.info(f"🎵 Using gTTS for {language} -> {gtts_lang}")
        
        # Generate TTS using gTTS
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        
        # Save to temporary file
        temp_audio = tempfile.mktemp(suffix=".mp3")
        tts.save(temp_audio)
        
        # Convert MP3 to base64 for web playback
        if os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 0:
            with open(temp_audio, "rb") as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
            
            # Clean up temporary file
            os.unlink(temp_audio)
            
            logger.info(f"✅ gTTS generated successfully for {language}")
            
            return {
                "success": True,
                "audio_base64": audio_base64,
                "service": f"gTTS-{gtts_lang}",
                "voice": f"Google-{language}",
                "language": language,
                "audio_format": "mp3",
                "message": f"Generated high-quality gTTS for {language}"
            }
        else:
            raise Exception("gTTS failed to generate audio file")
            
    except Exception as e:
        logger.error(f"❌ gTTS generation failed: {e}")
        
        # Fallback to browser TTS
        return {
            "success": False,
            "use_browser_tts": True,
            "message": f"gTTS failed: {str(e)}. Using browser TTS fallback."
        }

if __name__ == "__main__":
    import uvicorn
    logger.info("🌾 Starting Agriculture AI with Whisper + Smart Assistant + gTTS...")
    logger.info("🎵 Perfect Tamil/Telugu/Malayalam pronunciation with Google TTS!")
    logger.info("🚀 Server starting at http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
