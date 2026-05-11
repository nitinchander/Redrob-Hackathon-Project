import math

MAP = {
    "python": "python", "pyhton": "python", "java": "java", 
    "javascript": "javascript", "javascrpit": "javascript", "js": "javascript",
    "typescript": "typescript", "typescrpit": "typescript", "c++": "cpp", "cpp": "cpp",
    "r": "", "kotlin": "kotlin", "machinelearning": "machine_learning", 
    "machine learning": "machine_learning", "ml": "machine_learning", "sklearn": "machine_learning",
    "deeplearning": "deep_learning", "deep learning": "deep_learning", "deep-learning": "deep_learning",
    "tensorflow": "tensorflow", "pytorch": "pytorch", "keras": "keras", "nlp": "nlp",
    "bert": "bert", "xgboost": "xgboost", "feature engineering": "feature_engineering",
    "statistics": "statistics", "stats": "statistics", "regression": "regression",
    "clustering": "clustering", "data-viz": "data visualization", 
    "data visualization": "data visualization", "data viz": "data visualization",
    "matplotlib": "data_visualization", "tableau": "data visualization", 
    "power-bi": "data_visualization", "power bi": "data_visualization", 
    "powerbi": "data_visualization", "pandas": "pandas", "numpy": "numpy",
    "react": "react", "reacts": "react", "reactjs": "react", "vue": "vue", 
    "vue.js": "vue", "vuejs": "vue", "redux": "redux", "tailwind": "tailwind",
    "html/css": "html css", "html": "html css", "css": "html css", "jest": "jest",
    "graphql": "graphql", "node.js": "nodejs", "nodejs": "nodejs", "node js": "nodejs",
    "flask": "flask", "spring boot": "spring boot", "springboot": "spring_boot",
    "rest api": "rest api", "rest": "rest_api", "restapi": "rest_api", 
    "microservices": "microservices", "sql": "sql", "mysql": "mysql", "mysq": "mysql",
    "postgresql": "postgresql", "postgres": "postgresql", "mongodb": "mongodb",
    "redis": "redis", "docker": "docker", "kubernetes": "kubernetes", 
    "kubernates": "kubernetes", "k8s": "kubernetes", "ci/cd": "ci cd", 
    "cicd": "ci_cd", "aws": "aws", "android": "android", "firebase": "firebase",
    "algorithms": "algorithms", "algoritms": "algorithms", 
    "data structure": "data_structures", "data structures": "data_structures",
    "competitive programming": "competitive_programming", "ui/ux": "ui ux", "figma": "figma"
}

candidates = [
    {"name": "Arjun Sharma", "skills": "Pyhton, Machine Learning, SQL, pandas, numpy, Deep-learning"},
    {"name": "Priya Nair", "skills": "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"},
    {"name": "Rahul Gupta", "skills": "Java, Spring Boot, MySql, Microservices, Docker, kubernates"},
    {"name": "Sneha Patel", "skills": "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"},
    {"name": "Vikram Singh", "skills": "C++, Algoritms, Data Structure, competitive programming, python"},
    {"name": "Ananya Krishnan", "skills": "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"},
    {"name": "Karan Mehta", "skills": "Python, Sklearn, XGboost, feature engineering, SQL, tableau"},
    {"name": "Deepika Rao", "skills": "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"},
    {"name": "Aditya Kumar", "skills": "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"},
    {"name": "Meera Iyer", "skills": "python, R, statistics, ML, regression, clustering, Power-BI"}
]

jobs = [
    {"id": "JD-1", "title": "Kakao (ML Engineer)", "req": "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization, NLP, BERT, Feature Engineering, Statistics"},
    {"id": "JD-2", "title": "Naver (Backend Engineer)", "req": "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes, REST API, CI/CD, Redis"},
    {"id": "JD-3", "title": "Line (Frontend Engineer)", "req": "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS, Node.js, GraphQL, Redux, Jest, AWS"}
]

def get_normalized(raw_string):
    items = [i.strip().lower() for i in raw_string.split(',')]
    found = set()
    for item in items:
        if item in MAP and MAP[item]:
            found.add(MAP[item])
    return sorted(list(found))

for c in candidates: 
    c['clean'] = get_normalized(c['skills'])
for j in jobs: 
    j['clean'] = get_normalized(j['req'])

vocabulary = set()
for c in candidates:
    vocabulary.update(c['clean'])
vocab_list = sorted(list(vocabulary))

def get_tfidf(target_skills, full_list, vocab):
    n = len(target_skills)
    vector = []
    for s in vocab:
        if s in target_skills:
            tf = 1 / n
            df = sum(1 for c in full_list if s in c['clean'])
            idf = math.log(10 / df)
            vector.append(tf * idf)
        else:
            vector.append(0.0)
    return vector

for j in jobs:
    jd_vec = [1 if s in j['clean'] else 0 for s in vocab_list]
    scores = []
    
    for c in candidates:
        c_vec = get_tfidf(c['clean'], candidates, vocab_list)
        dot = sum(a * b for a, b in zip(c_vec, jd_vec))
        norm_a = math.sqrt(sum(a**2 for a in c_vec))
        norm_b = math.sqrt(sum(b**2 for b in jd_vec))
        
        sim = round(dot / (norm_a * norm_b), 2) if (norm_a * norm_b) > 0 else 0.0
        scores.append({"name": c['name'], "score": sim})
    
    top3 = sorted(scores, key=lambda x: (-x['score'], x['name']))[:3]
    
    print(f"{j['id']}\n{j['title']}")
    print(", ".join([f"{res['name']} ({res['score']:.2f})" for res in top3]))
    print("-" * 20)