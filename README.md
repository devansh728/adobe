### **After cloning the git repo before runnning docker file**

**In terminal**

1) git clone https://github.com/jhaaj08/Adobe-India-Hackathon25.git 
2) git clone https://github.com/facebookresearch/fastText.git   
3) cd fastText
4) pip install .  
5) cd ..
6) Start Building Docker file 

---

## 📚 **Detailed Documentation**

For comprehensive technical details and implementation specifics, please refer to:

### **Challenge 1A: PDF Outline Extraction**
- **[📋 README-1a.md](README-1a.md)** - Complete guide for Challenge 1A including architecture, Docker setup, and usage instructions

### **Challenge 1B: Persona-Driven Document Intelligence**  
- **[🔬 approach_explanation.md](approach_explanation.md)** - Detailed technical approach with flowcharts, code examples, and implementation details

---

## 🚀 **Quick Start**

### **Challenge 1A (PDF Outline Extraction)**
```bash
# Build Docker image
docker build --platform linux/amd64 -t challenge1a-processor .

# Run container
docker run --rm -v "$(pwd)/Adobe-India-Hackathon25/Challenge_1a/sample_dataset/outputs:/app/Adobe-India-Hackathon25/Challenge_1a/sample_dataset/outputs" challenge1a-processor
```

### **Challenge 1B (Persona-Driven Intelligence)**
```bash
# Build Docker image
docker build -f Dockerfile.1b -t challenge1b-processor .

# Run container
docker run --rm challenge1b-processor
```

For detailed setup instructions and technical implementation, see the documentation links above.


