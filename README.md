# Machine_Learning
## 🤖 Machine Learning Flowchart & Sub-Branches
## 🌐 What is Machine Learning?

Machine Learning (ML) is a subset of Artificial Intelligence (AI) that enables systems to learn from data and improve their performance over time without being explicitly programmed.
## 🏗️ ML Flowchart Overview

Below is a flowchart that shows the main branches of Machine Learning and their key categories:
~~~mermaid
flowchart TD
    A[Machine Learning] --> B[Supervised Learning]
    A --> C[Unsupervised Learning]
    A --> D[Reinforcement Learning]
    A --> E[Deep Learning]
    
    B --> B1[Regression]
    B --> B2[Classification]

    C --> C1[Clustering]
    C --> C2[Association]

    D --> D1[Markov Decision Process]
    D --> D2[Policy Optimization]

    E --> E1[Neural Networks]
    E --> E2[Convolutional Neural Networks - CNNs]
    E --> E3[Recurrent Neural Networks - RNNs]
~~~
## 🔎 Sub-Branches Explained

### 1. Supervised Learning
- **Definition**: Learning from labeled data (input → output).  
- **Subtypes**:  
  - **Regression** → Predicts continuous values (e.g., house prices, temperature).  
  - **Classification** → Predicts categories (e.g., spam vs. not spam, disease detection).  
- **Examples**: Predicting stock prices, image recognition.  

---

### 2. Unsupervised Learning
- **Definition**: Learning from unlabeled data by finding hidden patterns.  
- **Subtypes**:  
  - **Clustering** → Groups similar items (e.g., customer segmentation).  
  - **Association** → Finds relationships (e.g., “users who buy X also buy Y”).  
- **Examples**: Market basket analysis, grouping documents.  

---

### 3. Reinforcement Learning
- **Definition**: Learning by interacting with an environment and receiving rewards or penalties.  
- **Concepts**: Agent, Environment, Reward, Action, Policy.  
- **Subtypes**:  
  - **Markov Decision Process (MDP)** → Framework for sequential decision-making.  
  - **Policy Optimization** → Improves decision-making strategies over time.  
- **Examples**: Self-driving cars, robotics, game-playing AI (Chess, Go).  

---

### 4. Deep Learning
- **Definition**: A specialized subset of ML that uses artificial neural networks with multiple layers.  
- **Subtypes**:  
  - **Neural Networks (ANNs)** → General models for various tasks.  
  - **Convolutional Neural Networks (CNNs)** → Best for images and computer vision.  
  - **Recurrent Neural Networks (RNNs)** → Best for sequences (text, speech, time-series).  
- **Examples**: Face recognition, language translation, chatbots.  

---

## 🚀 Summary
- **Supervised Learning** → Labeled data → Regression & Classification  
- **Unsupervised Learning** → Unlabeled data → Clustering & Association  
- **Reinforcement Learning** → Learn by interaction → Rewards & Penalties  
- **Deep Learning** → Neural networks for images, speech, and NLP  
