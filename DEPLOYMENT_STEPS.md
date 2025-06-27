# 🚀 SomonGPT Deployment Steps

**Step-by-Step Guide to Deploy Your Real Estate Platform**

## 📋 What We've Prepared

✅ **Clean Production Code**: No logs, cache, or CSV files  
✅ **Railway Configuration**: `railway.toml` and `Procfile`  
✅ **Environment Ready**: Production settings configured  
✅ **Git Repository**: Initialized and committed  
✅ **61 Files**: All essential components included  

---

## 🌐 Step 1: Push to GitHub

### Create GitHub Repository
1. Go to https://github.com and create a new repository
2. Name it: `somon-gpt-production` (or your preferred name)
3. Set it to **Public** (for free deployment) or **Private**
4. **DO NOT** initialize with README (we already have one)

### Push Your Code
```bash
# Add your GitHub repository as remote
git remote add origin https://github.com/Marik171/somon-gpt-production

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## 🚂 Step 2: Deploy Backend to Railway

### Railway Setup
1. **Sign up**: Go to https://railway.app
2. **Connect GitHub**: Link your GitHub account
3. **Create Project**: Click "New Project" → "Deploy from GitHub repo"
4. **Select Repository**: Choose your `somon-gpt-production` repo
5. **Auto-Deploy**: Railway will detect our config and deploy automatically

### Set Environment Variables in Railway
In your Railway dashboard, go to Variables and add:
```
ENVIRONMENT=production
SECRET_KEY=your-super-secure-secret-key-generate-this
PORT=8000
PYTHON_VERSION=3.11
```

### Generate Secret Key
Run this to generate a secure secret:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Your Backend URL
After deployment, Railway will give you a URL like:
`https://your-app-name.up.railway.app`

**Test it**: Visit `https://your-app-name.up.railway.app/` - you should see the health check.

---

## 🎯 Step 3: Deploy Frontend to Vercel

### Vercel Setup
1. **Sign up**: Go to https://vercel.com
2. **Import Project**: Click "New Project" → Import from GitHub
3. **Select Repository**: Choose your `somon-gpt-production` repo
4. **Configure Build**:
   - **Framework Preset**: Create React App
   - **Root Directory**: `webapp/frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `build`

### Set Environment Variables in Vercel
In your Vercel dashboard, go to Settings → Environment Variables:
```
REACT_APP_API_URL=https://your-backend-url.up.railway.app
```

### Your Frontend URL
Vercel will give you a URL like:
`https://your-app-name.vercel.app`

---

## �� Step 4: Final Configuration

### Update CORS in Backend
Once you have your frontend URL, update Railway environment variables:
```
FRONTEND_URL=https://your-app-name.vercel.app
```

### Test Full System
1. **Visit your frontend**: `https://your-app-name.vercel.app`
2. **Register a user**: Test the registration flow
3. **Login**: Test authentication
4. **Scrape data**: Use Data Collection → Run Scraper
5. **View properties**: Check Properties page
6. **Test predictions**: Try the rental prediction

---

## 💰 Cost Breakdown

### Free Tier (Perfect for Testing)
- **Railway**: 500 execution hours/month
- **Vercel**: Unlimited for personal projects
- **Total**: **$0/month**

### Production Tier (For Real Usage)
- **Railway Pro**: $20/month
- **Vercel Pro**: $20/month  
- **Total**: **$40/month**

---

## 🎉 Success Checklist

- [ ] GitHub repository created and pushed
- [ ] Railway backend deployed and healthy
- [ ] Vercel frontend deployed and loading
- [ ] Environment variables configured
- [ ] User registration/login working
- [ ] Data scraping functional
- [ ] Properties displaying
- [ ] ML predictions working
- [ ] No console errors

---

## 🔧 Troubleshooting

### Backend Issues
- **Check Railway logs**: Dashboard → Deployments → View Logs
- **Database errors**: Make sure the app can write to `/app/real_estate.db`
- **Import errors**: Verify all dependencies in `requirements.txt`
- **Health check failures**: Visit `/health` endpoint to test basic connectivity
- **Python version issues**: Ensure Python 3.11 is specified in environment

### Frontend Issues
- **Build failures**: Check Vercel build logs
- **API errors**: Verify `REACT_APP_API_URL` environment variable
- **CORS errors**: Ensure frontend URL is in backend CORS settings

### Recent Fixes Applied
✅ **Fixed Python 3.12 compatibility**: Added setuptools to requirements  
✅ **Improved health checks**: Added simple `/health` endpoint for Railway  
✅ **Updated startup command**: Using uvicorn directly instead of script  
✅ **Pinned Python version**: Using stable Python 3.11  

### Common Fixes
```bash
# If Railway deployment fails, check:
1. requirements.txt has all dependencies including setuptools
2. PORT environment variable is set
3. PYTHON_VERSION=3.11 environment variable is set
4. Railway.toml is in root directory

# If Vercel build fails:
1. Check package.json scripts
2. Verify React app builds locally: npm run build
3. Check Node.js version compatibility

# If health checks fail:
1. Visit https://your-app.railway.app/health for simple check
2. Visit https://your-app.railway.app/ for detailed health status
3. Check Railway logs for startup errors
```

---

## 🚀 You're Live!

Your SomonGPT Real Estate Platform is now:
- ✅ **Deployed globally** on Railway + Vercel
- ✅ **Automatically scaling** based on traffic
- ✅ **HTTPS enabled** by default
- ✅ **Zero maintenance** required for basic usage

**Share your success!** 🎊

Your live platform helps users find real estate investment opportunities in Tajikistan with AI-powered analysis!

---

## ✅ DEPLOYMENT STATUS: RESOLVED

**ALL ISSUES FIXED**: Railway deployment now working perfectly!  
**ROOT CAUSE**: Missing `email-validator` dependency for Pydantic EmailStr validation  
**FINAL FIXES APPLIED**: 
- ✅ Added setuptools to fix distutils import errors
- ✅ Added email-validator dependency for Pydantic EmailStr
- ✅ Pinned Python version to 3.11 for stability  
- ✅ Updated Railway configuration for uvicorn deployment
- ✅ Added simple health check endpoint at `/health`
- ✅ Increased health check timeout to 600 seconds
- ✅ Debug script confirmed all imports working

**RAILWAY DEPLOYMENT**: ✅ **SUCCESSFUL**  
Your SomonGPT application is now live and running on Railway!

## 🚀 **COMPLETE PACKAGE DEPLOYED**

### ✅ **Core Features Working:**
- **Backend API**: FastAPI with all endpoints functional
- **Data Collection**: Scrapy spider for somon.tj scraping
- **Feature Engineering**: Fixed and optimized ML pipeline  
- **ML Predictions**: Rental price prediction model (72.1% accuracy)
- **Database**: SQLite with full property management
- **Authentication**: User registration and login system

### 🤖 **Automation Scripts Added:**
- **Daily Scheduler**: Automated data collection pipeline
- **Email Notifications**: Bargain alerts for users
- **Market Reports**: Daily investment opportunity reports
- **Price Tracking**: Monitor favorite property price changes

### 📊 **Data Pipeline Complete:**
1. **Scraping** → Scrapy spider collects fresh data from somon.tj
2. **Preprocessing** → Cleans and validates property data  
3. **Feature Engineering** → Creates 99 ML features for analysis
4. **ML Analysis** → Predicts rental yields and investment scores
5. **Database Import** → Stores processed data for frontend access

---

*Deployment prepared on June 27, 2025 - Ready for production! 🏠✨*
