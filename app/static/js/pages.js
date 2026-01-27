/**
 * Pages Script - Common functionality for Mypage, Survey
 */

document.addEventListener('DOMContentLoaded', function () {
    initTheme();
    initColorTheme();

    const themeToggle = document.getElementById('theme-toggle');
    const colorToggle = document.getElementById('color-toggle');

    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }

    if (colorToggle) {
        colorToggle.addEventListener('click', toggleColorTheme);
    }
});

function initTheme() {
    const savedTheme = localStorage.getItem('chatbot-theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('chatbot-theme', newTheme);
    updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.textContent = theme === 'dark' ? '☀️' : '🌙';
    }
}

function initColorTheme() {
    const savedColor = localStorage.getItem('chatbot-color') || 'gold';
    if (savedColor === 'green' || savedColor === 'brown') {
        document.documentElement.setAttribute('data-color', savedColor);
    }
    updateColorIcon(savedColor);
}

function toggleColorTheme() {
    const currentColor = document.documentElement.getAttribute('data-color') || 'gold';
    let newColor;

    if (currentColor === 'gold' || !currentColor) {
        newColor = 'green';
    } else if (currentColor === 'green') {
        newColor = 'brown';
    } else {
        newColor = 'gold';
    }

    if (newColor === 'gold') {
        document.documentElement.removeAttribute('data-color');
    } else {
        document.documentElement.setAttribute('data-color', newColor);
    }

    localStorage.setItem('chatbot-color', newColor);
    updateColorIcon(newColor);
}

function updateColorIcon(color) {
    const colorToggle = document.getElementById('color-toggle');
    if (colorToggle) {
        const icons = { gold: '🏆', green: '🌿', brown: '🍂' };
        colorToggle.textContent = icons[color] || '🏆';
    }
}
