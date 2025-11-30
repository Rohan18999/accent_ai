# AccentAI - Intelligent Accent Detection & Cuisine Recommendation System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green)
![HuBERT](https://img.shields.io/badge/Model-HuBERT-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-99.75%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

> An AI-powered web application that uses HuBERT self-supervised learning to detect regional Indian accents from English speech with 99.75% accuracy and recommends authentic regional cuisines based on the detected accent.

## 🎯 Features

- 🎤 **Real-time Voice Recording** - Browser-based audio capture with visual feedback
- 🧠 **AI-Powered Accent Detection** - 99.75% accuracy using HuBERT neural network
- 🍽️ **Personalized Cuisine Recommendations** - Authentic regional dishes based on detected accent
- 🌙 **Dark Mode Support** - Toggle between light and dark themes with persistent preference
- 📊 **Interactive UI** - Modern, responsive design with smooth animations
- 🔊 **Audio Playback** - Review your recording before analysis
- 📈 **Confidence Score** - Visual representation of prediction confidence
- 📱 **Responsive Design** - Works seamlessly on desktop and mobile devices

## 🌍 Supported Accents

| Region | Language | Sample Cuisines |
|--------|----------|-----------------|
| Andhra Pradesh | Telugu | Hyderabadi Biryani, Pulihora, Gongura Chutney |
| Gujarat | Gujarati | Dhokla, Thepla, Khandvi |
| Jharkhand | Hindi/Tribal | Litti Chokha, Dhuska, Rugra |
| Karnataka | Kannada | Mysore Masala Dosa, Bisi Bele Bath, Ragi Mudde |
| Kerala | Malayalam | Appam with Stew, Puttu, Avial |
| Tamil Nadu | Tamil | Chettinad Chicken, Sambar, Idli & Dosa |

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Modern web browser with microphone support

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/accent-cuisine-app.git
cd accent-cuisine-app
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Add your trained model**
```bash
# Place your hubert_layer_analysis.pth file in the models folder
cp /path/to/hubert_layer_analysis.pth models/
```

5. **Run the application**
```bash
python app.py
```

6. **Open in browser**
```
http://localhost:5001
```

## 📁 Project Structure

```
accent-cuisine-app/
│
├── app.py                  # Main Flask application
├── model_utils.py          # HuBERT model loading and prediction
├── cuisine_data.py         # Cuisine recommendations database
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
│
├── models/
│   └── hubert_layer_analysis.pth  # Trained HuBERT model (not in repo)
│
├── static/
│   ├── css/
│   │   └── style.css      # Main stylesheet with dark mode
│   ├── js/
│   │   └── recorder.js    # Audio recording and UI logic
│   └── uploads/           # Temporary audio storage
│
└── templates/
    └── index.html         # Main HTML template
```

## 🛠️ Technology Stack

### Backend
- **Flask 3.0.0** - Web framework
- **PyTorch 2.6+** - Deep learning framework
- **Transformers 4.35+** - HuBERT model from Hugging Face
- **Librosa 0.10+** - Audio processing
- **NumPy** - Numerical computations

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling with CSS Variables for theming
- **JavaScript (ES6+)** - Interactive functionality
- **Font Awesome 6.4** - Icons
- **Web Audio API** - Browser audio recording

### AI Model
- **HuBERT-base** (facebook/hubert-base-ls960)
- **Layer 3 Embeddings** - Optimal for accent detection
- **Self-supervised Learning** - Pre-trained on 960 hours of speech

## 📊 Model Performance

| Accent | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Andhra Pradesh | 1.00 | 0.99 | 0.99 | 269 |
| Gujarat | 1.00 | 1.00 | 1.00 | 45 |
| Jharkhand | 1.00 | 1.00 | 1.00 | 124 |
| Karnataka | 1.00 | 1.00 | 1.00 | 253 |
| Kerala | 1.00 | 1.00 | 1.00 | 251 |
| Tamil Nadu | 1.00 | 1.00 | 1.00 | 276 |
| **Overall** | **1.00** | **1.00** | **1.00** | **1218** |

**Overall Accuracy: 99.75%**

## 🎨 Usage

1. **Click the microphone button** on the homepage
2. **Speak a short English phrase** (e.g., "I would like to order food")
3. **Click "Stop Recording"** when finished
4. **Click "Analyze Now"** to detect your accent
5. **View results** - Your accent and personalized cuisine recommendations

### Example Phrases
- "I would like to order some food"
- "Can I see the menu please?"
- "What are today's specials?"

## 🔧 Configuration

### Change Server Port
Edit `app.py`:
```python
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)  # Change port here
```

### Specify HuBERT Layer
Edit `model_utils.py`:
```python
predictor = load_hubert_predictor(MODEL_PATH, best_layer=3)  # Specify layer
```

### Add New Cuisine
Edit `cuisine_data.py`:
```python
ACCENT_TO_CUISINE = {
    "new_region": {
        "region": "New Region",
        "native_language": "Language",
        "cuisines": [
            {"name": "Dish Name", "description": "Description"}
        ]
    }
}
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Samala Rohan Sidharth** - [GitHub](https://github.com/Rohan18999)

## 🙏 Acknowledgments

- **HuBERT Model** - Facebook AI Research
- **Dataset** - DarshanaS/IndicAccentDb from Hugging Face
- **Academic Institution** - JNTUH (Jawaharlal Nehru Technological University Hyderabad)
- **Inspiration** - Creating personalized dining experiences through AI

## 📧 Contact

For questions or feedback:
- GitHub: [@Rohan18999](https://github.com/Rohan18999)
- Email: your.email@example.com

## 🐛 Known Issues

- Audio recording requires HTTPS in production (use ngrok for local testing)
- Microphone permission required in browser
- Model file (~360MB) not included in repository

## 🔮 Future Enhancements

- [ ] Support for more Indian regional accents
- [ ] Multi-language support beyond English
- [ ] Integration with food delivery APIs
- [ ] Mobile app version (React Native)
- [ ] User accounts and history
- [ ] Social sharing features
- [ ] Restaurant recommendations

## 📚 Research References

- [HuBERT: Self-Supervised Speech Representation Learning](https://arxiv.org/abs/2106.07447)
- [IndicAccentDB Dataset](https://huggingface.co/datasets/DarshanaS/IndicAccentDb)

## 🎓 Academic Project

**Course**: AI/ML Specialization  
**Year**: 2025  
**Institution**: JNTUH  
**Project Type**: Full-Stack AI Application  

---

**⭐ If you found this project helpful, please give it a star!**

**🍽️ Happy Cuisine Discovery!**