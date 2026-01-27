/**
 * Survey Page - Psychological Assessment Script
 */

// Survey Questions
const questions = [
    {
        id: 1,
        text: "지난 2주 동안, 일상적인 활동에 대한 흥미나 즐거움이 줄어들었다고 느끼셨나요?",
        category: "depression"
    },
    {
        id: 2,
        text: "지난 2주 동안, 기분이 가라앉거나 우울하거나 희망이 없다고 느끼셨나요?",
        category: "depression"
    },
    {
        id: 3,
        text: "지난 2주 동안, 잠들기 어렵거나 자주 깨거나 너무 많이 주무셨나요?",
        category: "anxiety"
    },
    {
        id: 4,
        text: "지난 2주 동안, 피곤하거나 에너지가 부족하다고 느끼셨나요?",
        category: "stress"
    },
    {
        id: 5,
        text: "지난 2주 동안, 식욕이 줄었거나 과식을 하셨나요?",
        category: "stress"
    },
    {
        id: 6,
        text: "지난 2주 동안, 자신이 실패자라고 느끼거나 가족을 실망시켰다고 느끼셨나요?",
        category: "depression"
    },
    {
        id: 7,
        text: "지난 2주 동안, 집중하기 어려웠나요? (예: 신문을 읽거나 TV를 볼 때)",
        category: "anxiety"
    },
    {
        id: 8,
        text: "지난 2주 동안, 초조하거나 불안하거나 긴장되었나요?",
        category: "anxiety"
    },
    {
        id: 9,
        text: "지난 2주 동안, 걱정을 멈추거나 조절하기가 어려웠나요?",
        category: "anxiety"
    },
    {
        id: 10,
        text: "지난 2주 동안, 너무 많은 걱정 때문에 편히 쉬기가 어려웠나요?",
        category: "stress"
    }
];

let currentQuestion = 0;
let answers = [];

// Initialize survey
function startSurvey() {
    document.getElementById('survey-intro').style.display = 'none';
    document.getElementById('survey-questions').style.display = 'block';
    currentQuestion = 0;
    answers = [];
    showQuestion();
}

function showQuestion() {
    const question = questions[currentQuestion];

    document.getElementById('question-number').textContent = `Q${question.id}`;
    document.getElementById('question-text').textContent = question.text;
    document.getElementById('progress-text').textContent = `${currentQuestion + 1} / ${questions.length}`;
    document.getElementById('progress-fill').style.width = `${((currentQuestion + 1) / questions.length) * 100}%`;

    // Reset answer selection
    document.querySelectorAll('.answer-btn').forEach(btn => {
        btn.classList.remove('selected');
    });

    // Restore previous answer if exists
    if (answers[currentQuestion] !== undefined) {
        const selectedBtn = document.querySelector(`.answer-btn[data-value="${answers[currentQuestion]}"]`);
        if (selectedBtn) selectedBtn.classList.add('selected');
    }

    // Update navigation buttons
    document.getElementById('prev-btn').disabled = currentQuestion === 0;
    document.getElementById('next-btn').disabled = answers[currentQuestion] === undefined;
    document.getElementById('next-btn').textContent = currentQuestion === questions.length - 1 ? '결과 보기' : '다음';
}

function selectAnswer(btn) {
    document.querySelectorAll('.answer-btn').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    answers[currentQuestion] = parseInt(btn.getAttribute('data-value'));
    document.getElementById('next-btn').disabled = false;
}

function prevQuestion() {
    if (currentQuestion > 0) {
        currentQuestion--;
        showQuestion();
    }
}

function nextQuestion() {
    if (currentQuestion < questions.length - 1) {
        currentQuestion++;
        showQuestion();
    } else {
        showResult();
    }
}

function showResult() {
    document.getElementById('survey-questions').style.display = 'none';
    document.getElementById('survey-result').style.display = 'block';

    // Calculate scores
    const totalScore = answers.reduce((sum, val) => sum + val, 0);
    const maxScore = questions.length * 3;
    const percentage = Math.round(100 - (totalScore / maxScore) * 100);

    // Calculate category scores
    const categories = {
        depression: { total: 0, count: 0 },
        anxiety: { total: 0, count: 0 },
        stress: { total: 0, count: 0 }
    };

    questions.forEach((q, i) => {
        if (answers[i] !== undefined) {
            categories[q.category].total += answers[i];
            categories[q.category].count++;
        }
    });

    // Update result display
    const scoreCircle = document.getElementById('score-circle');
    document.getElementById('score-value').textContent = percentage;

    // Remove all score classes and add appropriate one
    scoreCircle.classList.remove('score-excellent', 'score-good', 'score-warning', 'score-danger');

    let resultIcon, resultTitle, advice, scoreClass;

    if (percentage >= 80) {
        resultIcon = '💚';
        resultTitle = '매우 양호';
        scoreClass = 'score-excellent';
        advice = '현재 매우 건강한 심리 상태를 유지하고 계십니다. 지금처럼 자기 관리를 잘 해주시면 됩니다. 가끔 스트레스 해소를 위한 취미 활동을 즐기시는 것도 좋습니다.';
    } else if (percentage >= 60) {
        resultIcon = '💛';
        resultTitle = '양호';
        scoreClass = 'score-good';
        advice = '전반적으로 양호한 상태입니다. 가벼운 스트레스 관리 기법을 배워보시는 것이 도움이 될 수 있습니다. 규칙적인 운동과 충분한 수면을 권장드립니다.';
    } else if (percentage >= 40) {
        resultIcon = '🧡';
        resultTitle = '주의 필요';
        scoreClass = 'score-warning';
        advice = '약간의 심리적 어려움이 있는 것으로 보입니다. AI 상담을 통해 현재 상황에 대해 이야기 나눠보시는 것을 권장드립니다. 필요시 전문가 상담도 고려해 보세요.';
    } else {
        resultIcon = '❤️';
        resultTitle = '전문 상담 권장';
        scoreClass = 'score-danger';
        advice = '현재 심리적으로 어려운 시기를 보내고 계신 것 같습니다. AI 상담과 함께 전문 상담사와의 상담을 강력히 권장드립니다. 혼자 힘들어하지 마시고 도움을 받으세요.';
    }

    scoreCircle.classList.add(scoreClass);
    document.getElementById('result-icon').textContent = resultIcon;
    document.getElementById('score-desc').textContent = resultTitle;
    document.getElementById('result-advice').textContent = advice;

    // Update detail bars
    const detailItems = document.querySelectorAll('.detail-item');
    const categoryNames = ['depression', 'anxiety', 'stress'];
    const categoryLabels = ['우울감', '불안감', '스트레스'];

    detailItems.forEach((item, i) => {
        const cat = categories[categoryNames[i]];
        const catPercentage = cat.count > 0 ? Math.round((cat.total / (cat.count * 3)) * 100) : 0;

        item.querySelector('.detail-fill').style.width = catPercentage + '%';

        let level;
        if (catPercentage <= 33) level = '낮음';
        else if (catPercentage <= 66) level = '보통';
        else level = '높음';

        item.querySelector('.detail-value').textContent = level;
    });
}

function restartSurvey() {
    document.getElementById('survey-result').style.display = 'none';
    document.getElementById('survey-intro').style.display = 'block';
    currentQuestion = 0;
    answers = [];
}
