from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import logging
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from pathlib import Path
from typing import Dict, List, Tuple
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

# Comprehensive Global Agriculture Knowledge Base
FALLBACK_KB = {
    "crops": {
        # Cereals/Grains
        "rice": {
            "en": "Rice is a staple grain crop. Best grown in flooded fields. Requires 4-6 months growing season. Plant during monsoon. Harvest when grains turn golden. Major varieties: Basmati, Jasmine, Arborio.",
            "ta": "அரிசி ஒரு முக்கிய தானிய பயிர். வெள்ளம் நிறைந்த வயல்களில் சிறப்பாக வளரும். 4-6 மாத வளர்ச்சி காலம் தேவை। பருவமழைக் காலத்தில் நடவு செய்யவும்।",
            "te": "వరి ప్రధాన ధాన్య పంట. నీరు నిండిన పొలాల్లో బాగా పెరుగుతుంది. 4-6 నెలల పెరుగుదల కాలం అవசరం.",
            "ml": "അരി ഒരു പ്രധാന ധാന്യ വിള. വെള്ളം നിറഞ്ഞ വയലുകളിൽ നന്നായി വളരും. 4-6 മാസത്തെ വളർച്ചാകാലം ആവശ്യം।",
            "hi": "चावल एक मुख्य अनाज की फसल है। बाढ़ वाले खेतों में सबसे अच्छी तरह उगता है। 4-6 महीने की बढ़ने की अवधि चाहिए।"
        },
        "wheat": {
            "en": "Wheat is a major cereal grain. Grows best in temperate climates. Sow in October-November. Harvest in March-April. Requires well-drained soil.",
            "ta": "கோதுமை ஒரு முக்கிய தானிய பயிர். மிதமான காலநிலையில் சிறப்பாக வளரும். அக்டோபர்-நவம்பரில் விதைக்கவும்।",
            "te": "గోధుమ ప్రధాన ధాన్య పంట. సమశీతోష్ణ వాతావరణంలో బాగా పెరుగుతుంది।",
            "ml": "ഗോതമ്പ് ഒരു പ്രധാന ധാന്യ വിള. മിതശീതോഷ്ണ കാലാവസ്ഥയിൽ നന്നായി വളരും।",
            "hi": "गेहूं एक मुख्य अनाज है। समशीतोष्ण जलवायु में सबसे अच्छी तरह उगता है।"
        },
        "corn": {
            "en": "Corn/Maize is a versatile cereal crop. Requires warm climate and well-drained soil. Plant after last frost. Harvest when kernels are milky. Used for food, feed, and industrial purposes.",
            "ta": "சோளம் ஒரு பல்நோக்கு தானிய பயிர். வெப்பமான காலநிலை மற்றும் நல்ல வடிகால் மண் தேவை।",
            "te": "మొక్కజొన్న బహుళ ఉపయోగ ధాన్య పంట. వెచ్చని వాతావరణం మరియు మంచి డ్రైనేజ్ అవసరం।",
            "ml": "ചോളം ഒരു ബഹുമുഖ ധാന്യ വിള. ചൂടുള്ള കാലാവസ്ഥയും നല്ല ഡ്രെയിനേജും ആവശ്യം।",
            "hi": "मक्का एक बहुउपयोगी अनाज की फसल है। गर्म जलवायु और अच्छी जल निकासी चाहिए।"
        },
        "barley": {
            "en": "Barley is a hardy cereal grain. Tolerates cool, dry conditions. Plant in fall or spring. Used for brewing, animal feed, and food.",
            "ta": "பார்லி ஒரு கடினமான தானிய பயிர். குளிர், வறண்ட நிலைமைகளை தாங்கும்।",
            "te": "బార్లీ దృఢమైన ధాన్య పంట. చల్లని, పొడి పరిస్థితులను తట్టుకోగలదు।",
            "ml": "ബാർലി കാഠിന്യമുള്ള ധാന്യ വിള. തണുത്ത, വരണ്ട അവസ്ഥകൾ സഹിക്കും।",
            "hi": "जौ एक कठोर अनाज है। ठंडी, सूखी परिस्थितियों को सहन करता है।"
        },
        "oats": {
            "en": "Oats are cool-season cereal grain. Prefer cooler climates. Plant in early spring or fall. Good for human consumption and animal feed.",
            "ta": "ஓட்ஸ் குளிர்கால தானிய பயிர். குளிர்ந்த காலநிலையை விரும்பும்।",
            "te": "వోట్స్ చల్లని కాలం ధాన్య పంట. చల్లని వాతావరణాన్ని ఇష్టపడుతుంది।",
            "ml": "ഓട്സ് തണുത്ത കാലാവസ്ഥയിലെ ധാന്യ വിള. തണുത്ത കാലാവസ്ഥ ഇഷ്ടപ്പെടുന്നു।",
            "hi": "जई ठंडे मौसम का अनाज है। ठंडी जलवायु पसंद करता है।"
        },
        # Legumes/Pulses
        "soybeans": {
            "en": "Soybeans are protein-rich legumes. Require warm growing season. Plant after soil warms. Fix nitrogen in soil. Harvest when pods rattle.",
            "ta": "சோயாபீன் புரதம் நிறைந்த பருப்பு வகை। வெப்பமான வளர்ச்சி காலம் தேவை।",
            "te": "సోయాబీన్స్ ప్రోటీన్ అధికంగా ఉండే గింజలు. వెచ్చని పెరుగుదల కాలం అవసరం।",
            "ml": "സോയാബീൻ പ്രോട്ടീൻ സമ്പുഷ്ടമായ പയർവർഗ്ഗം. ചൂടുള്ള വളർച്ചാ കാലാവസ്ഥ ആവശ്യം।",
            "hi": "सोयाबीन प्रोटीन युक्त दलहन है। गर्म बढ़ने का मौसम चाहिए।"
        },
        "chickpeas": {
            "en": "Chickpeas are drought-tolerant legumes. Prefer cool, dry conditions. Plant in winter/spring. Fix nitrogen. Good protein source.",
            "ta": "கொண்டைக்கடலை வறட்சியை தாங்கும் பருப்பு வகை। குளிர், வறண்ட நிலைமைகளை விரும்பும்।",
            "te": "శనగలు కరువు తట్టుకునే గింజలు. చల్లని, పొడి పరిస్థితులను ఇష్టపడతాయి।",
            "ml": "ചെറുപയർ വരൾച്ച സഹിക്കുന്ന പയർവർഗ്ഗം. തണുത്ത, വരണ്ട അവസ്ഥകൾ ഇഷ്ടപ്പെടുന്നു।",
            "hi": "चना सूखा सहने वाली दाल है। ठंडी, सूखी परिस्थितियां पसंद करता है।"
        },
        "lentils": {
            "en": "Lentils are cool-season legumes. Tolerate frost. Plant in fall or early spring. Quick-growing protein crop. Various colors available.",
            "ta": "பருப்பு குளிர்கால பருப்பு வகை। உறைபனியை தாங்கும். இலையுதிர் அல்லது வசந்த காலத்தில் நடவு।",
            "te": "మసూర్ చల్లని కాలం గింజలు. మంచును తట్టుకుంటాయి।",
            "ml": "പയർ തണുത്ത കാലാവസ്ഥയിലെ പയർവർഗ്ഗം. തുഷാരം സഹിക്കും।",
            "hi": "मसूर ठंडे मौसम की दाल है। पाला सहन करती है।"
        },
        # Vegetables
        "tomatoes": {
            "en": "Tomatoes are warm-season vegetables. Need support structures. Require consistent watering. Harvest when fully colored but firm.",
            "ta": "தக்காளி வெப்பகால காய்கறி। ஆதார கட்டமைப்பு தேவை। நிலையான நீர்ப்பாசனம் தேவை।",
            "te": "టమోటాలు వేసవి కాలపు కూరగాయలు. మద్దతు నిర్మాణాలు అవసరం।",
            "ml": "തക്കാളി വേനൽക്കാല പച്ചക്കറി. പിന്തുണ ഘടനകൾ ആവശ്യം।",
            "hi": "टमाटर गर्म मौसम की सब्जी है। सहारे की संरचना चाहिए।"
        },
        "potatoes": {
            "en": "Potatoes are cool-season tubers. Plant in early spring. Hill soil around plants. Harvest when tops die back. Store in cool, dark place.",
            "ta": "உருளைக்கிழங்கு குளிர்கால கிழங்கு வகை। வசந்த காலத்தின் ஆரம்பத்தில் நடவு செய்யவும்।",
            "te": "బంగాళాదుంపలు చల్లని కాలం గడ్డకంద. వసంత ఋతువు ప్రారంభంలో నాటాలి।",
            "ml": "ഉരുളക്കിഴങ്ങ് തണുത്ത കാലാവസ്ഥയിലെ കിഴങ്ങ്. വസന്തത്തിന്റെ തുടക്കത്തിൽ നടുക।",
            "hi": "आलू ठंडे मौसम का कंद है। वसंत की शुरुआत में लगाएं।"
        },
        "onions": {
            "en": "Onions are biennial bulbs grown as annuals. Prefer cool weather for growth, warm weather for bulbing. Long day vs short day varieties.",
            "ta": "வெங்காயம் இரு ஆண்டு பயிராக வளர்க்கப்படும் ஆண்டு பயிர். வளர்ச்சிக்கு குளிர் காலநிலையை விரும்பும்।",
            "te": "ఉల్లిపాయలు రెండేళ్ల బల్బులు వార్షిక పంటలుగా పెరుగుతాయి। పెరుగుదలకు చల్లని వాతావరణం కావాలి।",
            "ml": "ഉള്ളി വാർഷിക വിളയായി വളർത്തുന്ന ദ്വിവാർഷിക ബൾബുകൾ. വളർച്ചയ്ക്ക് തണുത്ത കാലാവസ്ഥ ഇഷ്ടപ്പെടുന്നു।",
            "hi": "प्याज द्विवार्षिक बल्ब हैं जो वार्षिक फसल के रूप में उगाए जाते हैं।"
        },
        "carrots": {
            "en": "Carrots are cool-season root vegetables. Need loose, deep soil. Direct seed in garden. Thin seedlings. Harvest when roots reach desired size.",
            "ta": "கேரட் குளிர்கால வேர் காய்கறி। தளர்வான, ஆழமான மண் தேவை। தோட்டத்தில் நேரடி விதை।",
            "te": "క్యారెట్లు చల్లని కాలపు వేరు కూరగాయలు. వదులుగా, లోతైన మట్టి అవసరం।",
            "ml": "കാരറ്റ് തണുത്ത കാലാവസ്ഥയിലെ വേര് പച്ചക്കറി. അയഞ്ഞതും ആഴമുള്ളതുമായ മണ്ണ് ആവശ്യം।",
            "hi": "गाजर ठंडे मौसम की जड़ सब्जी है। ढीली, गहरी मिट्टी चाहिए।"
        },
        # Fruits
        "apples": {
            "en": "Apples are temperate fruit trees. Require chill hours in winter. Plant in spring. Need cross-pollination. Harvest in fall when ripe.",
            "ta": "ஆப்பிள் மிதமான பழ மரங்கள். குளிர்காலத்தில் குளிர் மணி நேரம் தேவை। வசந்த காலத்தில் நடவு।",
            "te": "ఆపిల్స్ సమశీతోష్ణ పండ్ల చెట్లు. శీతాకాలంలో చల్లని గంటలు అవసరం।",
            "ml": "ആപ്പിൾ മിതശീതോഷ്ണ ഫലവൃക്ഷങ്ങൾ. ശീതകാലത്ത് തണുത്ത മണിക്കൂറുകൾ ആവശ്യം।",
            "hi": "सेब समशीतोष्ण फलों के पेड़ हैं। सर्दी में ठंडे घंटे चाहिए।"
        },
        "oranges": {
            "en": "Oranges are citrus fruits. Need warm, frost-free climate. Regular watering required. Harvest when fully colored and sweet.",
            "ta": "ஆரஞ்சு சிட்ரஸ் பழங்கள். வெப்பமான, உறைபனி இல்லாத காலநிலை தேவை।",
            "te": "నారింజలు సిట్రస్ పండ్లు. వెచ్చని, మంచు లేని వాతావరణం అవసరం।",
            "ml": "ഓറഞ്ച് സിട്രസ് ഫലങ്ങൾ. ചൂടുള്ളതും തുഷാരരഹിതവുമായ കാലാവസ്ഥ ആവശ്യം।",
            "hi": "संतरे खट्टे फल हैं। गर्म, पाला रहित जलवायु चाहिए।"
        },
        "bananas": {
            "en": "Bananas are tropical fruits. Need hot, humid climate. Require rich, well-drained soil. Harvest bunches when plump but green.",
            "ta": "வாழைப்பழம் வெப்பமண்டல பழங்கள். வெப்பமான, ஈரப்பதமான காலநிலை தேவை।",
            "te": "అరటిపండ్లు ఉష్ణమండల పండ్లు. వేడిమిగిలిన, తేమతో కూడిన వాతావరణం అవసరం।",
            "ml": "വാഴപ്പഴം ഉഷ്ണമേഖലാ ഫലങ്ങൾ. ചൂടുള്ളതും ഈർപ്പമുള്ളതുമായ കാലാവസ്ഥ ആവശ്യം।",
            "hi": "केले उष्णकटिबंधीय फल हैं। गर्म, नम जलवायु चाहिए।"
        },
        "grapes": {
            "en": "Grapes are perennial vines. Need warm, dry growing season. Require trellising support. Harvest when sugar content is optimal.",
            "ta": "திராட்சை பல ஆண்டு கொடிகள். வெப்பமான, வறண்ட வளர்ச்சி காலம் தேவை।",
            "te": "ద్రాక్షలు బహుఏళ్ల తీగలు. వెచ్చని, పొడి పెరుగుదల కాలం అవసరం।",
            "ml": "മുന്തിരി ബഹുവാർഷിക വള്ളികൾ. ചൂടുള്ളതും വരണ്ടതുമായ വളർച്ചാ കാലം ആവശ്യം।",
            "hi": "अंगूर बारहमासी बेल हैं। गर्म, सूखा बढ़ने का मौसम चाहिए।"
        },
        # Cash Crops
        "cotton": {
            "en": "Cotton is a warm-season fiber crop. Requires long, hot growing season. Deep, well-drained soil needed. Harvest when bolls open.",
            "ta": "பருத்தி வெப்பகால நார் பயிர். நீண்ட, வெப்பமான வளர்ச்சி காலம் தேவை।",
            "te": "పత్తి వేసవి కాలపు నారు పంట. సుదీర్ఘమైన, వేడిమిగిలిన పెరుగుదల కాలం అవసరం।",
            "ml": "പരുത്തി വേനൽക്കാല നാര് വിള. ദീർഘവും ചൂടുള്ളതുമായ വളർച്ചാ കാലം ആവശ്യം।",
            "hi": "कपास गर्म मौसम की रेशा फसल है। लंबा, गर्म बढ़ने का मौसम चाहिए।"
        },
        "sugarcane": {
            "en": "Sugarcane is a tropical cash crop. Requires hot, humid climate. 12-18 month crop cycle. Needs abundant water. Harvest when stalks are mature.",
            "ta": "கரும்பு வெப்பமண்டல பணப் பயிர். வெப்பமான, ஈரப்பதமான காலநிலை தேவை।",
            "te": "చెరకు ఉష్ణమండల వాణిజ్య పంట. వేడిమిగిలిన, తేమతో కూడిన వాతావరణం అవసరం।",
            "ml": "കരിമ്പ് ഉഷ്ണമേഖലാ വാണിജ്യ വിള. ചൂടുള്ളതും ഈർപ്പമുള്ളതുമായ കാലാവസ്ഥ ആവശ്യം।",
            "hi": "गन्ना उष्णकटिबंधीय नकदी फसल है। गर्म, नम जलवायु चाहिए।"
        },
        "coffee": {
            "en": "Coffee is a tropical perennial shrub. Needs high altitude, consistent rainfall. Shade-grown preferred. Harvest cherries when ripe.",
            "ta": "காபி வெப்பமண்டல பல ஆண்டு புதர். உயர் பகுதி, நிலையான மழை தேவை।",
            "te": "కాఫీ ఉష్ణమండల బహుఏళ్ల పొద. ఎత్తైన ప్రాంతం, స్థిరమైన వర్షపాతం అవసరం।",
            "ml": "കാപ്പി ഉഷ്ണമേഖലാ ബഹുവാർഷിക കുറ്റിച്ചെടി. ഉയർന്ന പ്രദേശം, സ്ഥിരമായ മഴ ആവശ്യം।",
            "hi": "कॉफी उष्णकटिबंधीय बारहमासी झाड़ी है। ऊंचाई, लगातार बारिश चाहिए।"
        },
        "tea": {
            "en": "Tea is a perennial evergreen shrub. Prefers cool, misty climate. Well-drained acidic soil needed. Harvest young leaves regularly.",
            "ta": "தேயிலை பல ஆண்டு பசுமையான புதர். குளிர், மூடுபனி காலநிலையை விரும்பும்।",
            "te": "తేనీరు బహుఏళ్ల సతత హరిత పొద. చల్లని, పొగమంచుతో కూడిన వాతావరణాన్ని ఇష్టపడుతుంది।",
            "ml": "ചായ ബഹുവാർഷിക നിത്യഹരിത കുറ്റിച്ചെടി. തണുത്തതും മൂടിക്കെട്ടിയതുമായ കാലാവസ്ഥ ഇഷ്ടപ്പെടുന്നു।",
            "hi": "चाय बारहमासी सदाबहार झाड़ी है। ठंडी, धुंधली जलवायु पसंद करती है।"
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

# Load Knowledge Base (external JSON if present; fallback to built-in)
DATA_PATH = Path(__file__).parent / "data" / "agri_kb.json"

def load_kb() -> Dict[str, Dict]:
    if DATA_PATH.exists():
        try:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                kb = json.load(f)
            logger.info(f"📚 Loaded external KB: {DATA_PATH}")
            return kb
        except Exception as e:
            logger.warning(f"Could not load external KB ({e}), using built-in fallback.")
    return FALLBACK_KB

AGRICULTURE_KB = load_kb()

# Multilingual, character n-gram based RAG index with crop/soil boosting
kb_index: Dict[str, Dict] = {}

def build_lang_index(lang: str):
    texts: List[str] = []
    keys: List[Tuple[str, str]] = []

    for category, items in AGRICULTURE_KB.items():
        for item, langs in items.items():
            content = (langs.get(lang) or langs.get('en') or '').strip()
            if content:
                texts.append(content)
                keys.append((category, item))

    if not texts:
        # fallback to English entries if none for the language
        for category, items in AGRICULTURE_KB.items():
            for item, langs in items.items():
                content = (langs.get('en') or '').strip()
                if content:
                    texts.append(content)
                    keys.append((category, item))

    vect = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        min_df=1,
        max_features=5000
    )
    mat = vect.fit_transform(texts)

    kb_index[lang] = {
        'vectorizer': vect,
        'vectors': mat,
        'keys': keys,
        'texts': texts
    }
    logger.info(f"✅ RAG index built for '{lang}' with {len(texts)} entries")

def ensure_lang_index(lang: str):
    if lang not in kb_index:
        build_lang_index(lang)

def detect_explicit_crop(query: str, language: str) -> str:
    """Detect crop mentions in multiple languages"""
    crop_synonyms = {
        'en': {
            # Cereals
            'rice': 'rice', 'paddy': 'rice', 'wheat': 'wheat', 'corn': 'corn', 'maize': 'corn',
            'barley': 'barley', 'oats': 'oats', 'millet': 'millet', 'quinoa': 'quinoa',
            # Legumes
            'soybeans': 'soybeans', 'soy': 'soybeans', 'chickpeas': 'chickpeas', 'lentils': 'lentils',
            'beans': 'beans', 'peas': 'peas', 'groundnut': 'groundnut', 'peanut': 'groundnut',
            # Vegetables
            'tomato': 'tomatoes', 'tomatoes': 'tomatoes', 'potato': 'potatoes', 'potatoes': 'potatoes',
            'onion': 'onions', 'onions': 'onions', 'carrot': 'carrots', 'carrots': 'carrots',
            'cabbage': 'cabbage', 'lettuce': 'lettuce', 'spinach': 'spinach', 'broccoli': 'broccoli',
            # Fruits
            'apple': 'apples', 'apples': 'apples', 'orange': 'oranges', 'oranges': 'oranges',
            'banana': 'bananas', 'bananas': 'bananas', 'grape': 'grapes', 'grapes': 'grapes',
            'mango': 'mango', 'papaya': 'papaya', 'pineapple': 'pineapple',
            # Cash crops
            'cotton': 'cotton', 'sugarcane': 'sugarcane', 'coffee': 'coffee', 'tea': 'tea',
            'tobacco': 'tobacco', 'rubber': 'rubber'
        },
        'ta': {
            # Tamil crop names
            'அரிசி': 'rice', 'நெல்': 'rice', 'கோதுமை': 'wheat', 'சோளம்': 'corn',
            'கேழ்வரகு': 'millet', 'பார்லி': 'barley', 'வெண்ணையடுங்': 'barley',
            'சோயாபீன்': 'soybeans', 'கொண்டைக்கடலை': 'chickpeas', 'பருப்பு': 'lentils',
            'தக்காளி': 'tomatoes', 'உருளைக்கிழங்கு': 'potatoes', 'வெங்காயம்': 'onions',
            'கேரட்': 'carrots', 'முட்டைக்கோஸ்': 'cabbage', 'கீரை': 'spinach',
            'ஆப்பிள்': 'apples', 'ஆரஞ்சு': 'oranges', 'வாழைப்பழம்': 'bananas',
            'திராட்சை': 'grapes', 'மாம்பழம்': 'mango', 'பப்பாளி': 'papaya',
            'பருத்தி': 'cotton', 'கரும்பு': 'sugarcane', 'காபி': 'coffee', 'தேயிலை': 'tea'
        },
        'te': {
            # Telugu crop names
            'వరి': 'rice', 'బియ్యం': 'rice', 'గోధుమ': 'wheat', 'మొక్కజొన్న': 'corn',
            'జొన్న': 'millet', 'బార్లీ': 'barley', 'వోట్స్': 'oats',
            'సోయాబీన్స్': 'soybeans', 'శనగలు': 'chickpeas', 'మసూర్': 'lentils',
            'టమోటా': 'tomatoes', 'బంగాళాదుంప': 'potatoes', 'ఉల్లిపాయలు': 'onions',
            'క్యారెట్': 'carrots', 'కాబేజీ': 'cabbage', 'పాలకూర': 'spinach',
            'ఆపిల్స్': 'apples', 'నారింజలు': 'oranges', 'అరటిపండ్లు': 'bananas',
            'ద్రాక్షలు': 'grapes', 'మామిడిపండు': 'mango', 'బొప్పాయి': 'papaya',
            'పత్తి': 'cotton', 'చెరకు': 'sugarcane', 'కాఫీ': 'coffee', 'తేనీరు': 'tea'
        },
        'ml': {
            # Malayalam crop names
            'അരി': 'rice', 'നെൽ': 'rice', 'ഗോതമ്പ്': 'wheat', 'ചോളം': 'corn',
            'കേഴ്വരകു': 'millet', 'ബാർലി': 'barley', 'ഓട്സ്': 'oats',
            'സോയാബീൻ': 'soybeans', 'ചെറുപയർ': 'chickpeas', 'പയർ': 'lentils',
            'തക്കാളി': 'tomatoes', 'ഉരുളക്കിഴങ്ങ്': 'potatoes', 'ഉള്ളി': 'onions',
            'കാരറ്റ്': 'carrots', 'കാബേജ്': 'cabbage', 'ചീര': 'spinach',
            'ആപ്പിൾ': 'apples', 'ഓറഞ്ച്': 'oranges', 'വാഴപ്പഴം': 'bananas',
            'മുന്തിരി': 'grapes', 'മാമ്പഴം': 'mango', 'പപ്പായ': 'papaya',
            'പരുത്തി': 'cotton', 'കരിമ്പ്': 'sugarcane', 'കാപ്പി': 'coffee', 'ചായ': 'tea'
        },
        'hi': {
            # Hindi crop names
            'चावल': 'rice', 'धान': 'rice', 'गेहूं': 'wheat', 'मक्का': 'corn',
            'बाजरा': 'millet', 'जौ': 'barley', 'जई': 'oats',
            'सोयाबीन': 'soybeans', 'चना': 'chickpeas', 'मसूर': 'lentils',
            'टमाटर': 'tomatoes', 'आलू': 'potatoes', 'प्याज': 'onions',
            'गाजर': 'carrots', 'पत्तागोभी': 'cabbage', 'पालक': 'spinach',
            'सेब': 'apples', 'संतरा': 'oranges', 'केला': 'bananas',
            'अंगूर': 'grapes', 'आम': 'mango', 'पपीता': 'papaya',
            'कपास': 'cotton', 'गन्ना': 'sugarcane', 'कॉफी': 'coffee', 'चाय': 'tea'
        }
    }
    
    mapping = crop_synonyms.get(language, crop_synonyms['en'])
    query_lower = query.lower()
    
    for crop_name, standardized_name in mapping.items():
        if crop_name.lower() in query_lower:
            return standardized_name
    
    return ""

def is_agriculture_related(query: str, language: str) -> bool:
    """Check if query is agriculture-related and reject non-agricultural queries"""
    agriculture_keywords = {
        'en': [
            'crop', 'farming', 'agriculture', 'plant', 'soil', 'fertilizer', 'pest', 'disease',
            'irrigation', 'harvest', 'seed', 'growth', 'cultivation', 'farm', 'field',
            'pesticide', 'herbicide', 'organic', 'yield', 'planting', 'sowing', 'tractor',
            'compost', 'manure', 'greenhouse', 'nursery', 'pruning', 'grafting', 'weather',
            'climate', 'rain', 'drought', 'water', 'nitrogen', 'phosphorus', 'potassium',
            'ph', 'acidity', 'alkaline', 'mulch', 'weeds', 'insects', 'fungus', 'bacteria',
            'rice', 'wheat', 'corn', 'barley', 'oats', 'tomato', 'potato', 'onion', 'carrot',
            'apple', 'orange', 'banana', 'grape', 'cotton', 'sugarcane', 'coffee', 'tea',
            'beans', 'peas', 'lentil', 'soybean', 'cabbage', 'lettuce', 'spinach', 'mango'
        ],
        'ta': [
            'பயிர்', 'விவசாயம்', 'வேளாண்மை', 'தாவரம்', 'மண்', 'உரம்', 'பூச்சி', 'நோய்',
            'நீர்ப்பாசனம்', 'அறுவடை', 'விதை', 'வளர்ச்சி', 'சாகுபடி', 'வயல்', 'அரிசி', 'நெல்',
            'கோதுமை', 'தக்காளி', 'உருளைக்கிழங்கு', 'வெங்காயம்', 'கேரட்', 'ஆப்பிள்', 'பருத்தி',
            'பூச்சிக்கொல்லி', 'களைக்கொல்லி', 'கரிம', 'விளைச்சல்', 'நடவு', 'கரும்பு', 'காபி'
        ],
        'te': [
            'పంట', 'వ్యవసాయం', 'వేషధారణ', 'మొక్క', 'మట్టి', 'ఎరువులు', 'కీటకాలు', 'వ్యాధి',
            'నీటిపారుదల', 'కోత', 'విత్తనం', 'పెరుగుదల', 'సాగు', 'పొలం', 'వరి', 'గోధుమ',
            'టమోటా', 'బంగాళాదుంప', 'ఉల్లిపాయలు', 'క్యారెట్', 'ఆపిల్స్', 'పత్తి', 'చెరకు'
        ],
        'ml': [
            'വിള', 'കൃഷി', 'കാർഷികം', 'ചെടി', 'മണ്ണ്', 'വളം', 'കീടം', 'രോഗം',
            'ജലസേചനം', 'വിളവെടുപ്പ്', 'വിത്ത്', 'വളർച്ച', 'കൃഷി', 'വയൽ', 'അരി', 'ഗോതമ്പ്',
            'തക്കാളി', 'ഉരുളക്കിഴങ്ങ്', 'ഉള്ളി', 'കാരറ്റ്', 'ആപ്പിൾ', 'പരുത്തി', 'കരിമ്പ്'
        ],
        'hi': [
            'फसल', 'खेती', 'कृषि', 'पौधा', 'मिट्टी', 'खाद', 'कीट', 'बीमारी',
            'सिंचाई', 'कटाई', 'बीज', 'वृद्धि', 'खेती', 'खेत', 'चावल', 'गेहूं',
            'टमाटर', 'आलू', 'प्याज', 'गाजर', 'सेब', 'कपास', 'गन्ना'
        ]
    }
    
    # First check for crop names (most important)
    crop_detected = detect_explicit_crop(query, language)
    if crop_detected:
        return True
    
    # Get keywords for the language
    keywords = agriculture_keywords.get(language, agriculture_keywords['en'])
    query_lower = query.lower()
    
    # Check if any agriculture keyword is present
    for keyword in keywords:
        if keyword.lower() in query_lower:
            return True
    
    # Additional check for common agriculture phrases
    agri_phrases = {
        'en': ['grow', 'plant', 'farm', 'soil', 'water', 'sun', 'season', 'harvest', 'food production', 'agricultural'],
        'ta': ['வளர்', 'நட', 'பயிர்', 'உணவு', 'மழை', 'காலநிலை'],
        'te': ['పెరుగు', 'నాట', 'పంట', 'ఆహారం', 'వర్షం', 'వాతావరణం'],
        'ml': ['വളർ', 'നട', 'വിള', 'ഭക്ഷണം', 'മഴ', 'കാലാവസ്ഥ'],
        'hi': ['उग', 'लगा', 'फसल', 'भोजन', 'बारिश', 'मौसम']
    }
    
    phrases = agri_phrases.get(language, agri_phrases['en'])
    for phrase in phrases:
        if phrase.lower() in query_lower:
            return True
    
    # If nothing matches, it's probably not agriculture-related
    return False

class QueryRequest(BaseModel):
    query: str
    language: str = "en"
    mode: str = "direct"
    user_type: str = "farmer"  # farmer, expert, student
    crop_type: str = ""
    land_size: str = ""
    soil_type: str = ""

def get_rag_context(query: str, language: str = "en", top_k: int = 3, user_crop: str = "", user_soil: str = ""):
    try:
        ensure_lang_index(language)
        idx = kb_index[language]
        vect = idx['vectorizer']
        mat = idx['vectors']
        keys = idx['keys']

        query_vector = vect.transform([query])
        similarities = cosine_similarity(query_vector, mat)[0]

        # Boost by selected crop/soil
        crop_boost = 0.25 if user_crop else 0.0
        soil_boost = 0.15 if user_soil else 0.0
        for i, (cat, item) in enumerate(keys):
            if user_crop and cat == 'crops' and item.lower() == user_crop.lower():
                similarities[i] += crop_boost
            if user_soil and cat == 'soil' and item.lower() == user_soil.lower():
                similarities[i] += soil_boost

        top_indices = similarities.argsort()[-top_k:][::-1]

        relevant_context = []
        for idx_i in top_indices:
            if similarities[idx_i] <= 0:
                continue
            category, item = keys[idx_i]
            context_data = AGRICULTURE_KB[category][item]
            content = context_data.get(language) or context_data.get('en', '')
            relevant_context.append({
                'category': category,
                'item': item,
                'content': content,
                'similarity': float(similarities[idx_i])
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
                <strong>🎯 Enhanced Global Agriculture Assistant:</strong><br>
                🎤 Whisper: Accurate speech recognition in any language<br>
                🤖 Smart Assistant: COMPREHENSIVE support for ALL crops worldwide<br>
                🔊 gTTS: High-quality Google TTS for Indian languages<br>
                🌾 Global Crops: Cereals, Legumes, Vegetables, Fruits, Cash Crops<br>
                ⚡ <strong>ALL CROPS SUPPORTED! Clean answers in all languages!</strong>
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
                        <option value="rice">🌾 Rice/Paddy</option>
                        <option value="wheat">🌾 Wheat</option>
                        <option value="corn">🌽 Corn/Maize</option>
                        <option value="barley">🌾 Barley</option>
                        <option value="oats">🌾 Oats</option>
                        <option value="soybeans">🫘 Soybeans</option>
                        <option value="chickpeas">🫘 Chickpeas</option>
                        <option value="lentils">🫘 Lentils</option>
                        <option value="tomatoes">🍅 Tomatoes</option>
                        <option value="potatoes">🥔 Potatoes</option>
                        <option value="onions">🧅 Onions</option>
                        <option value="carrots">🥕 Carrots</option>
                        <option value="apples">🍎 Apples</option>
                        <option value="oranges">🍊 Oranges</option>
                        <option value="bananas">🍌 Bananas</option>
                        <option value="grapes">🍇 Grapes</option>
                        <option value="cotton">🌿 Cotton</option>
                        <option value="sugarcane">🎋 Sugarcane</option>
                        <option value="coffee">☕ Coffee</option>
                        <option value="tea">🍵 Tea</option>
                        <option value="vegetables">🥬 Mixed Vegetables</option>
                        <option value="fruits">🍎 Mixed Fruits</option>
                        <option value="other">🌱 Other Crop</option>
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
                    <button class="quick-question" onclick="askQuickQuestion('How much water does my crop need and when?')">
                        💧 Water requirements and timing
                    </button>
                    <button class="quick-question" onclick="askQuickQuestion('What fertilizer should I use for maximum yield?')">
                        🧪 Best fertilizers for high yield
                    </button>
                    <button class="quick-question" onclick="askQuickQuestion('How to prevent and treat crop diseases?')">
                        🦠 Disease prevention and treatment
                    </button>
                    <button class="quick-question" onclick="askQuickQuestion('When is the best time to plant my crop?')">
                        📅 Optimal planting seasons
                    </button>
                    <button class="quick-question" onclick="askQuickQuestion('How to control pests naturally?')">
                        🐛 Natural pest control methods
                    </button>
                    <button class="quick-question" onclick="askQuickQuestion('What are the signs of nutrient deficiency?')">
                        📊 Nutrient deficiency symptoms
                    </button>
                    <button class="quick-question" onclick="askQuickQuestion('How to improve soil quality?')">
                        🌍 Soil improvement techniques
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
                Welcome to Enhanced Global Agriculture AI! 🌾<br><br>
                <strong>🌍 COMPREHENSIVE CROP SUPPORT:</strong><br>
                � Cereals: Rice, Wheat, Corn, Barley, Oats, Millet<br>
                🫘 Legumes: Soybeans, Chickpeas, Lentils, Beans, Peas<br>
                🥬 Vegetables: Tomatoes, Potatoes, Onions, Carrots, Cabbage<br>
                🍎 Fruits: Apples, Oranges, Bananas, Grapes, Mango<br>
                🌿 Cash Crops: Cotton, Sugarcane, Coffee, Tea, Tobacco<br><br>
                <strong>�️ Agriculture-Only Assistant:</strong><br>
                ✅ Answers ALL agriculture-related questions<br>
                🌐 Clean answers in ALL languages<br>
                🎤 Whisper + 🤖 Smart RAG + 🔊 gTTS = Perfect combo!
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
                
                const status = `🟢 Ready | Global Crop Assistant + Whisper + gTTS | Language: ${langNames[selectedLanguage]}`;
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
                console.log('🚀 askAI function called with Enhanced Global Agriculture Assistant');
                
                const query = document.getElementById('queryInput').value;
                const responseDiv = document.getElementById('response');
                const details = getUserDetails();

                if (!query.trim()) {
                    alert('Please enter a question about agriculture or farming!');
                    return;
                }

                console.log(`📝 Query: "${query}" in language: ${selectedLanguage}`);
                responseDiv.innerHTML = '<div class="loading">🌾 Global Agriculture Assistant analyzing your question with RAG...</div>';

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
                    console.log('✅ Received Enhanced Agriculture Assistant response:', data);
                    
                    let responseText = `<strong>🌾 Global Agriculture Assistant Response:</strong>\\n\\n${data.answer}`;
                    
                    if (data.rag_sources && data.rag_sources.length > 0) {
                        responseText += `\\n\\n📚 <strong>Knowledge Sources:</strong>\\n`;
                        data.rag_sources.forEach((source, idx) => {
                            responseText += `${idx + 1}. ${source.category.toUpperCase()}: ${source.item}\\n`;
                        });
                    }
                    
                    if (data.restriction) {
                        responseText += `\\n\\n🛡️ <strong>Note:</strong> This assistant only responds to agriculture-related questions.`;
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

            // Allow Enter key to submit
            document.getElementById('queryInput').addEventListener('keypress', function(event) {
                if (event.key === 'Enter') {
                    askAI();
                }
            });

            console.log('🚀 Enhanced Global Agriculture AI ready with comprehensive crop support!');
        </script>
    </body>
    </html>
    """

@app.post("/query")
async def query_agriculture(request: QueryRequest):
    start_time = time.time()
    
    try:
        logger.info(f"🌾 Smart RAG Query: {request.query[:50]}... | Language: {request.language} | Profile: {request.user_type}")
        
        # Get RAG context for agriculture query (removed restriction filter)
        explicit_crop = detect_explicit_crop(request.query, request.language)
        rag_context = get_rag_context(
            request.query,
            request.language,
            top_k=3,
            user_crop=explicit_crop or request.crop_type,
            user_soil=request.soil_type
        )
        
        # Enhanced Smart Agriculture Assistant with comprehensive crop knowledge
        def generate_smart_agriculture_answer(query, language, rag_context, user_context):
            """Generate practical agriculture answers with comprehensive crop support"""
            
            # Build comprehensive answer from RAG context
            if rag_context:
                best_match = rag_context[0]
                answer = best_match['content']
                
                # Add specific advice based on user context and crop type
                crop_type = user_context.get('crop_type', '')
                soil_type = user_context.get('soil_type', '')
                land_size = user_context.get('land_size', '')
                
                # Soil-specific advice
                if soil_type == 'clay':
                    if language == 'ta':
                        answer += " களிமண் மண்ணுக்கு: நல்ல வடிகால் அமைப்பு அவசியம். கரிமப் பொருட்களைச் சேர்க்கவும்."
                    elif language == 'te':
                        answer += " మట్టి మట్టికి: మంచి డ్రైనేజ్ వ్యవస్థ అవసరం. సేంద్రీయ పదార్థాలను కలపండి."
                    elif language == 'ml':
                        answer += " കളിമണ്ണിന്: നല്ല ഡ്രെയിനേജ് സിസ്റ്റം ആവശ്യം. ജൈവവസ്തുക്കൾ ചേർക്കുക."
                    elif language == 'hi':
                        answer += " चिकनी मिट्टी के लिए: अच्छी जल निकासी व्यवस्था जरूरी। जैविक पदार्थ मिलाएं।"
                    else:
                        answer += " For clay soil: Good drainage system essential. Add organic matter to improve structure."
                
                elif soil_type == 'sandy':
                    if language == 'ta':
                        answer += " மணல் மண்ணுக்கு: அடிக்கடி நீர்ப்பாசனம் தேவை। கம்போஸ்ட் சேர்த்து ஊட்டச்சத்து தக்கவைக்கவும்."
                    elif language == 'te':
                        answer += " ఇసుక మట్టికి: తరచుగా నీటిపారుదల అవసరం. కంపోస్ట్ చేర்చి పోషకాలను నిలుపుకోండి."
                    elif language == 'ml':
                        answer += " മണൽമണ്ണിന്: ഇടയ്ക്കിടെ നനയ്ക്കണം. കമ്പോസ്റ്റ് ചേർത്ത് പോഷകങ്ങൾ നിലനിർത്തുക."
                    elif language == 'hi':
                        answer += " रेतीली मिट्टी के लिए: बार-बार सिंचाई चाहिए। कंपोस्ट मिलाकर पोषक तत्व बनाए रखें।"
                    else:
                        answer += " For sandy soil: Frequent irrigation needed. Add compost to retain nutrients."
                
                # Land size specific advice
                if land_size == 'small':
                    if language == 'ta':
                        answer += " சிறிய நிலத்திற்கு: சொட்டு நீர்ப்பாசனம், செங்குத்து விவசாயம், மண்ணின் மல்ச்சிங் பரிந்துரைக்கப்படுகிறது."
                    elif language == 'te':
                        answer += " చిన్న భూమికి: డ్రిప్ నీటిపారుదల, నిలువు వ్యవసాయం, మట్టి కవరింగ్ సిఫార్సు చేయబడింది."
                    elif language == 'ml':
                        answer += " ചെറിய ഭൂമിക്ക്: ഡ്രിപ്പ് ജലസേചനം, ലംബമായ കൃഷി, മണ്ണ് മൾച്ചിംഗ് ശുപാർശ ചെയ്യുന്നു."
                    elif language == 'hi':
                        answer += " छोटी जमीन के लिए: ड्रिप सिंचाई, ऊर्ध्वाधर खेती, मिट्टी मल्चिंग की सिफारिश।"
                    else:
                        answer += " For small land: Drip irrigation, vertical farming, soil mulching recommended."
                
                # Seasonal advice
                import datetime
                current_month = datetime.datetime.now().month
                
                if 6 <= current_month <= 10:  # Monsoon/Kharif season
                    if language == 'ta':
                        answer += " தற்போது கரீப் பருவம். அரிசி, பருத்தி, சோளம், கரும்பு ஆகியவற்றுக்கு ஏற்ற காலம்."
                    elif language == 'te':
                        answer += " ప్రస్తుతం ఖరీఫ్ సీజన్. వరి, పత్తి, మొక్కజొన్న, చెరకుకు అనువైన సమయం."
                    elif language == 'ml':
                        answer += " ഇപ്പോൾ ഖരീഫ് സീസൺ. അരി, പരുത്തി, ചോളം, കരിമ്പ് എന്നിവയ്ക്ക് അനുയോജ്യമായ സമയം."
                    elif language == 'hi':
                        answer += " अभी खरीफ मौसम। धान, कपास, मक्का, गन्ने के लिए उपयुक्त समय।"
                    else:
                        answer += " Current Kharif season. Suitable time for rice, cotton, corn, sugarcane."
                
                elif 11 <= current_month <= 4:  # Winter/Rabi season
                    if language == 'ta':
                        answer += " தற்போது ரபி பருவம். கோதுமை, பார்லி, கடுகு, பட்டாணி ஆகியவற்றுக்கு ஏற்ற காலம்."
                    elif language == 'te':
                        answer += " ప్రస్తుతం రబీ సీజన్. గోధుమ, బార్లీ, ఆవాలు, బఠానుల కోసం అనువైన సమయం."
                    elif language == 'ml':
                        answer += " ഇപ്പോൾ റബീ സീസൺ. ഗോതമ്പ്, ബാർലി, കടുക്, പയർ എന്നിവയ്ക്ക് അനുയോജ്യമായ സമയം."
                    elif language == 'hi':
                        answer += " अभी रबी मौसम। गेहूं, जौ, सरसों, मटर के लिए उपयुक्त समय।"
                    else:
                        answer += " Current Rabi season. Suitable time for wheat, barley, mustard, peas."
                
                return answer
            
            # Enhanced fallback responses with comprehensive crop support
            practical_fallbacks = {
                'en': {
                    'general': "For general farming: 1) Test soil pH (6.0-7.5 optimal), 2) Use organic compost, 3) Follow proper irrigation schedule, 4) Monitor for pests. I can help with any crop - cereals (rice, wheat, corn), legumes (soybeans, chickpeas), vegetables (tomatoes, potatoes), fruits (apples, oranges), or cash crops (cotton, sugarcane).",
                    'fertilizer': "Balanced NPK fertilizer guide: Most crops need 40kg Urea + 25kg DAP + 15kg MOP per acre. Split application - half at sowing, rest after 30-45 days. Organic options: compost, vermicompost, green manure.",
                    'pest': "Integrated pest management: 1) Neem oil spray (5ml/liter), 2) Remove affected parts, 3) Yellow sticky traps, 4) Beneficial insects, 5) Crop rotation. Specific treatments vary by crop and pest type.",
                    'disease': "Disease prevention: 1) Proper spacing for air circulation, 2) Avoid overhead watering, 3) Remove infected parts immediately, 4) Copper-based fungicides for fungal issues, 5) Resistant varieties when available."
                },
                'ta': {
                    'general': "பொதுவான வேளாண்மைக்கு: 1) மண் pH சோதனை (6.0-7.5 சிறந்தது), 2) கரிம கம்போஸ்ட் பயன்படுத்தவும், 3) சரியான நீர்ப்பாசனம், 4) பூச்சிகள் கண்காணிப்பு. நான் அனைத்து பயிர்களுக்கும் உதவ முடியும் - தானியங்கள், பருப்பு வகைகள், காய்கறிகள், பழங்கள், பணப்பயிர்கள்.",
                    'fertilizer': "சமச்சீர் NPK உரம்: பெரும்பாலான பயிர்களுக்கு ஏக்கருக்கு 40கிலோ யூரியா + 25கிலோ DAP + 15கிலோ MOP. பிரித்த பயன்பாடு - பாதி விதைக்கும்போது, மீதம் 30-45 நாட்களுக்குப் பிறகு.",
                    'pest': "ஒருங்கிணைந்த பூச்சி மேலாண்மை: 1) வேப்ப எண்ணெய் தெளிப்பு, 2) பாதிக்கப்பட்ட பகுதிகளை அகற்றவும், 3) மஞ்சள் ஒட்டும் பொறிகள், 4) பயன்படை பூச்சிகள், 5) பயிர் சுழற்சி.",
                    'disease': "நோய் தடுப்பு: 1) காற்றோட்டத்திற்கு சரியான இடைவெளி, 2) இலைகளில் நேரடி நீர் தெளிப்பு தவிர்க்கவும், 3) பாதிக்கப்பட்ட பகுதிகளை உடனே அகற்றவும்."
                },
                'te': {
                    'general': "సాధారణ వ్యవసాయానికి: 1) మట్టి pH పరీక్ష (6.0-7.5 ఉత్తమం), 2) సేంద్రీయ కంపోస్ట్ ఉపయోగించండి, 3) సరైన నీటిపారుదల, 4) కీటకాల పర్యవేక్షణ. నేను అన్ని పంటలకు సహాయం చేయగలను.",
                    'fertilizer': "సమతుల్య NPK ఎరువు: చాలా పంటలకు ఎకరకు 40కిలో యూరియా + 25కిలో DAP + 15కిలో MOP. విభజిత అప్లికేషన్ - సగం విత్తనాలో, మిగిలింది 30-45 రోజుల తర్వాత.",
                    'pest': "సమగ్ర కీటక నిర్వహణ: 1) వేప నూనె స్ప్రే, 2) ప్రభావిత భాగాలను తొలగించండి, 3) పసుపు జిగురు ఉచ్చులు, 4) ప్రయోజనకరమైన కీటకాలు.",
                    'disease': "వ్యాధి నివారణ: 1) గాలి ప్రసరణ కోసం సరైన అంతరం, 2) పై నుండి నీరు పోయడం మానుకోండి, 3) సోకిన భాగాలను వెంటనే తొలగించండి."
                },
                'ml': {
                    'general': "പൊതുവായ കൃഷിക്ക്: 1) മണ്ണിന്റെ pH പരിശോധന (6.0-7.5 ഉത്തമം), 2) ജൈവ കമ്പോസ്റ്റ് ഉപയോഗിക്കുക, 3) ശരിയായ ജലസേചനം, 4) കീടങ്ങളുടെ നിരീക്ഷണം. എനിക്ക് എല്ലാ വിളകൾക്കും സഹായിക്കാൻ കഴിയും.",
                    'fertilizer': "സമതുലിതമായ NPK വളം: മിക്ക വിളകൾക്കും ഏക്കറിന് 40കിലോ യൂറിയ + 25കിലോ DAP + 15കിലോ MOP. വിഭജിത പ്രയോഗം - പകുതി വിതയ്ക്കുമ്പോൾ, ബാക്കി 30-45 ദിവസങ്ങൾക്ക് ശേഷം.",
                    'pest': "സംയോജിത കീട പരിപാലനം: 1) വേപ്പെണ്ണ സ്പ്രേ, 2) ബാധിത ഭാഗങ്ങൾ നീക്കം ചെയ്യുക, 3) മഞ്ഞ ഒട്ടുന്ന കെണികൾ.",
                    'disease': "രോഗ പ്രതിരോധം: 1) വായു സഞ്ചാരത്തിന് ശരിയായ അകലം, 2) മുകളിൽ നിന്ന് വെള്ളം ഒഴിക്കുന്നത് ഒഴിവാക്കുക, 3) രോഗബാധിത ഭാഗങ്ങൾ ഉടനെ നീക്കം ചെയ്യുക."
                },
                'hi': {
                    'general': "सामान्य कृषि के लिए: 1) मिट्टी pH जांच (6.0-7.5 आदर्श), 2) जैविक खाद का उपयोग, 3) उचित सिंचाई, 4) कीट निगरानी। मैं सभी फसलों के लिए मदद कर सकता हूं।",
                    'fertilizer': "संतुलित NPK उर्वरक: अधिकांश फसलों के लिए प्रति एकड़ 40किलो यूरिया + 25किलो DAP + 15किलो MOP। विभाजित उपयोग - आधा बुआई के समय, बाकी 30-45 दिन बाद।",
                    'pest': "एकीकृत कीट प्रबंधन: 1) नीम तेल स्प्रे, 2) प्रभावित भागों को हटाएं, 3) पीले चिपचिपे जाल, 4) लाभकारी कीड़े।",
                    'disease': "रोग रोकथाम: 1) हवा के संचार के लिए उचित दूरी, 2) ऊपर से पानी देना बचें, 3) संक्रमित भागों को तुरंत हटाएं।"
                }
            }
            
            # Determine response category based on query analysis
            query_lower = query.lower()
            if any(word in query_lower for word in ['fertilizer', 'urea', 'dap', 'nutrients', 'উরम্', 'ఎరువులు', 'വളം', 'खाद']):
                category = 'fertilizer'
            elif any(word in query_lower for word in ['pest', 'insect', 'bug', 'spray', 'পূচ্চি', 'కీటకాలు', 'കീടം', 'कीट']):
                category = 'pest'
            elif any(word in query_lower for word in ['disease', 'fungus', 'rot', 'blight', 'নোয়', 'వ్యాధి', 'രോഗം', 'बीमारी']):
                category = 'disease'
            else:
                category = 'general'
            
            return practical_fallbacks.get(language, practical_fallbacks['en']).get(category, practical_fallbacks['en']['general'])
        
        # Generate comprehensive answer
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
            "confidence": 0.95,
            "processing_time_ms": round(processing_time),
            "language": request.language,
            "mode": "comprehensive_agriculture_assistant",
            "model": "Enhanced Smart RAG with Global Crop Support",
            "rag_sources": [{"category": ctx['category'], "item": ctx['item'], "similarity": ctx['similarity']} for ctx in rag_context],
            "user_context": user_context,
            "supported_crops": "All global crops supported including cereals, legumes, vegetables, fruits, cash crops"
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        
        # Enhanced fallback with agriculture focus
        practical_fallback = {
            'en': "I'm your comprehensive agriculture assistant for ALL crops worldwide. Ask me about: Cereals (rice, wheat, corn, barley), Legumes (soybeans, chickpeas, lentils), Vegetables (tomatoes, potatoes, onions), Fruits (apples, oranges, bananas), Cash crops (cotton, sugarcane, coffee). I provide soil, pest, fertilizer, and growing advice!",
            'ta': "நான் உலகளாவிய அனைத்து பயிர்களுக்கும் விரிவான வேளாண் உதவியாளர். என்னிடம் கேளுங்கள்: தானியங்கள், பருப்பு வகைகள், காய்கறிகள், பழங்கள், பணப்பயிர்கள் பற்றி.",
            'te': "నేను ప్రపంచవ్యాప్త అన్ని పంటలకు సమగ్ర వ్యవసాయ సహాయకుడను. నన్ను అడగండి: ధాన్యాలు, గింజలు, కూరగాయలు, పండ్లు, వాణిజ్య పంటల గురించి.",
            'ml': "ഞാൻ ലോകമെമ്പാടുമുള്ള എല്ലാ വിളകൾക്കും സമഗ്ര കാർഷിക സഹായിയാണ്. എന്നോട് ചോദിക്കുക: ധാന്യങ്ങൾ, പയർവർഗ്ഗങ്ങൾ, പച്ചക്കറികൾ, ഫലങ്ങൾ, വാണിജ്യ വിളകൾ.",
            'hi': "मैं दुनिया भर की सभी फसलों के लिए व्यापक कृषि सहायक हूं। मुझसे पूछें: अनाज, दालें, सब्जियां, फल, नकदी फसलों के बारे में।"
        }
        
        return {
            "answer": practical_fallback.get(request.language, practical_fallback['en']),
            "confidence": 0.8,
            "processing_time_ms": round((time.time() - start_time) * 1000),
            "language": request.language,
            "mode": "enhanced_fallback",
            "model": "Global Crop Assistant",
            "error": "handled_gracefully"
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
    import os
    
    # Get port from environment variable for Railway deployment
    port = int(os.environ.get("PORT", 8002))
    
    logger.info("🌾 Starting Enhanced Global Agriculture AI with Whisper + Smart Assistant + gTTS...")
    logger.info("🎵 Perfect Tamil/Telugu/Malayalam pronunciation with Google TTS!")
    logger.info(f"🚀 Server starting on port {port}")
    
    # For production deployment, bind to all interfaces
    uvicorn.run(app, host="0.0.0.0", port=port)
