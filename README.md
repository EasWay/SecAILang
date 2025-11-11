# 🏛️ Welfare Secretary AI

An intelligent AI assistant for welfare committee management, built with Flask and Google Gemini AI. Generate professional reports, manage committee data, and access insights from anywhere.

## ✨ Features

- 🤖 **AI-Powered Responses** - Intelligent answers about welfare committee activities
- 📊 **Report Generation** - Professional PDF and Word document reports
- 📱 **Mobile-Friendly** - Gemini-style responsive interface
- 📁 **Data Management** - Easy Excel file uploads via web interface
- 🚀 **Auto-Deployment** - GitHub Actions for seamless updates
- 🔄 **Real-time Updates** - Update data without redeploying
- 💾 **Automatic Backups** - Scheduled data backups
- 🔍 **Health Monitoring** - Automated uptime checks

## 🚀 Quick Start

### Local Development

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/welfare-secretary-ai.git
cd welfare-secretary-ai
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
# Create .env file
echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
```

4. **Run the application:**
```bash
python app.py
```

5. **Access the app:**
- Main interface: http://localhost:5000
- Admin panel: http://localhost:5000/admin

### Cloud Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions using GitHub Actions.

## 📋 Usage

### Chat Interface
- Ask questions about welfare committee activities
- Request comprehensive reports
- Get financial summaries and event details
- Download responses as PDF or Word documents

### Admin Panel
- Upload new Excel files to update data
- Monitor data status and statistics
- Manage committee information without redeployment

### Example Queries
- "Generate a comprehensive welfare report"
- "What are the total finances collected?"
- "Tell me about recent events"
- "What meetings have been held?"

## 🛠️ Technology Stack

- **Backend:** Flask, Python
- **AI:** Google Gemini Pro, LangChain
- **Data:** Pandas, FAISS Vector Store
- **Documents:** ReportLab (PDF), python-docx (Word)
- **Frontend:** HTML, CSS, JavaScript (Gemini UI style)
- **Deployment:** GitHub Actions, Railway/Render/Heroku
- **Data Storage:** Excel files, Vector embeddings

## 📁 Project Structure

```
welfare-secretary-ai/
├── app.py                 # Main Flask application
├── templates/
│   ├── index.html        # Main chat interface
│   └── admin.html        # Admin panel
├── .github/workflows/    # GitHub Actions
│   ├── deploy.yml        # Deployment workflow
│   ├── update-data.yml   # Data update workflow
│   ├── scheduled-backup.yml # Backup workflow
│   └── health-check.yml  # Health monitoring
├── requirements.txt      # Python dependencies
├── SECRETARY FORM(1-6).xlsx # Sample data file
├── .env                  # Environment variables (local)
└── README.md            # This file
```

## 🔧 Configuration

### Environment Variables
- `GOOGLE_API_KEY` - Your Google Gemini API key
- `PORT` - Server port (default: 5000)

### GitHub Secrets (for deployment)
- `GOOGLE_API_KEY` - Google API key
- `RAILWAY_TOKEN` - Railway deployment token
- `RAILWAY_SERVICE_ID` - Railway service ID
- `RAILWAY_APP_URL` - App URL for health checks

## 📊 Data Format

The Excel file should contain columns for:
- Committee member information
- Financial data (collected, spent, remaining amounts)
- Event details (name, location, attendance, outcomes)
- Meeting information (agenda, decisions)
- Issues and comments

## 🔄 Updating Data

### Method 1: Web Interface
1. Go to `/admin` on your deployed app
2. Upload new Excel file
3. Data is automatically processed and updated

### Method 2: GitHub Repository
1. Replace Excel file in repository
2. Commit and push changes
3. GitHub Actions automatically validates and redeploys

## 🚀 Deployment Options

- **Railway** (Recommended) - Easy setup, generous free tier
- **Render** - Simple deployment, good free tier
- **Heroku** - Reliable, paid plans available
- **Streamlit Cloud** - Alternative for Streamlit version

## 🔍 Monitoring

- **Health Checks** - Automated every 30 minutes
- **Data Backups** - Daily scheduled backups
- **Deployment Status** - GitHub Actions dashboard
- **Error Logging** - Comprehensive error tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- Check [DEPLOYMENT.md](DEPLOYMENT.md) for deployment help
- Review GitHub Actions logs for troubleshooting
- Test locally before deploying
- Ensure Excel file format matches expected structure

## 🎯 Roadmap

- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Email report scheduling
- [ ] Integration with Google Sheets
- [ ] Mobile app version
- [ ] Advanced user authentication

---

Built with ❤️ for welfare committee management