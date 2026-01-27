/**
 * FileName    : auth.js
 * Auth        : 박수빈
 * Date        : 2026-01-05
 * Description : 로그인/회원가입 페이지 스크립트
 *               Bcrypt 암호화 및 세션 기반 인증 연동
 * Issue/Note  : 다음 우편번호 API 연동 포함
 */

document.addEventListener('DOMContentLoaded', function () {
    // =============================================================
    // 초기화
    // =============================================================

    initColorTheme();
    initPasswordToggle();
    initPasswordStrength();
    initPasswordConfirmation();
    initLoginForm();
    initSignupForm();
    initUsernameCheck();
    initAddressSearch();

    // =============================================================
    // 테마 초기화
    // =============================================================

    /**
     * localStorage에서 저장된 테마 불러오기
     */
    function initColorTheme() {
        const savedTheme = localStorage.getItem('chatbot-theme') || 'light';
        const savedColor = localStorage.getItem('chatbot-color') || 'gold';

        document.documentElement.setAttribute('data-theme', savedTheme);
        if (savedColor === 'green' || savedColor === 'brown') {
            document.documentElement.setAttribute('data-color', savedColor);
        }
    }

    // =============================================================
    // 비밀번호 표시/숨김 토글
    // =============================================================

    /**
     * 비밀번호 입력창 표시/숨김 버튼 이벤트 핸들러
     */
    function initPasswordToggle() {
        document.querySelectorAll('.toggle-password').forEach(btn => {
            btn.addEventListener('click', function () {
                const input = this.parentElement.querySelector('input');
                const type = input.type === 'password' ? 'text' : 'password';
                input.type = type;

                // 아이콘 업데이트
                this.innerHTML = type === 'text' ? `
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path>
                        <line x1="1" y1="1" x2="23" y2="23"></line>
                    </svg>
                ` : `
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                        <circle cx="12" cy="12" r="3"></circle>
                    </svg>
                `;
            });
        });
    }

    // =============================================================
    // 비밀번호 강도 표시
    // =============================================================

    /**
     * 비밀번호 입력 시 강도 바 업데이트
     */
    function initPasswordStrength() {
        const passwordInput = document.getElementById('password');
        const strengthBar = document.querySelector('.strength-bar');

        if (passwordInput && strengthBar) {
            passwordInput.addEventListener('input', function () {
                const password = this.value;
                let strength = 0;

                // ---------------------------------------------------------
                // 강도 계산: 길이 8자 이상, 소문자, 대문자, 숫자/특수문자
                // ---------------------------------------------------------
                if (password.length >= 8) strength += 25;
                if (password.match(/[a-z]/)) strength += 25;
                if (password.match(/[A-Z]/)) strength += 25;
                if (password.match(/[0-9]/) || password.match(/[^a-zA-Z0-9]/)) strength += 25;

                strengthBar.style.width = strength + '%';
            });
        }
    }

    // =============================================================
    // 비밀번호 확인
    // =============================================================

    /**
     * 비밀번호 확인 입력 시 일치 여부 검사
     */
    function initPasswordConfirmation() {
        const passwordInput = document.getElementById('password');
        const passwordConfirm = document.getElementById('password-confirm');

        if (passwordConfirm && passwordInput) {
            passwordConfirm.addEventListener('input', function () {
                const formGroup = this.closest('.form-group');
                if (this.value && this.value !== passwordInput.value) {
                    formGroup.classList.add('error');
                    formGroup.classList.remove('success');
                } else if (this.value && this.value === passwordInput.value) {
                    formGroup.classList.remove('error');
                    formGroup.classList.add('success');
                } else {
                    formGroup.classList.remove('error', 'success');
                }
            });
        }
    }

    // =============================================================
    // 로그인 폼 처리
    // =============================================================

    /**
     * 로그인 폼 제출 이벤트 핸들러
     * API 호출하여 실제 로그인 처리
     */
    function initLoginForm() {
        const loginForm = document.getElementById('login-form');
        if (loginForm) {
            loginForm.addEventListener('submit', async function (e) {
                e.preventDefault();

                const username = document.getElementById('username').value;
                const password = document.getElementById('password').value;

                try {
                    // ---------------------------------------------------------
                    // 로그인 API 호출
                    // ---------------------------------------------------------
                    const response = await fetch('/api/login', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ username, password })
                    });

                    const data = await response.json();

                    if (data.success) {
                        // 로그인 성공 - 세션 데이터 저장 (localStorage에도 백업)
                        if (data.user) {
                            localStorage.setItem('user', JSON.stringify(data.user));
                        }
                        showToast('로그인 성공! 🎉');
                        setTimeout(() => {
                            window.location.href = data.redirect || '/';
                        }, 1000);
                    } else {
                        showToast(data.message || '로그인에 실패했습니다');
                    }
                } catch (error) {
                    console.error('Login error:', error);
                    showToast('로그인 중 오류가 발생했습니다');
                }
            });
        }
    }

    // =============================================================
    // 회원가입 폼 처리
    // =============================================================

    /**
     * 회원가입 폼 제출 이벤트 핸들러
     * 모든 필드 유효성 검사 후 API 호출
     */
    function initSignupForm() {
        const signupForm = document.getElementById('signup-form');
        if (signupForm) {
            signupForm.addEventListener('submit', async function (e) {
                e.preventDefault();

                // ---------------------------------------------------------
                // 폼 데이터 수집
                // ---------------------------------------------------------
                const username = document.getElementById('username').value;
                const password = document.getElementById('password').value;
                const passwordConfirm = document.getElementById('password-confirm').value;
                const name = document.getElementById('name').value;
                const gender = document.querySelector('input[name="gender"]:checked')?.value || '';
                const birthdate = document.getElementById('birthdate').value;
                const phone = document.getElementById('phone').value;
                const address = document.getElementById('address').value;
                const addressDetail = document.getElementById('address-detail').value;

                // ---------------------------------------------------------
                // 유효성 검사
                // ---------------------------------------------------------
                if (password !== passwordConfirm) {
                    showToast('비밀번호가 일치하지 않습니다');
                    return;
                }

                if (!window.usernameChecked) {
                    showToast('아이디 중복확인을 해주세요');
                    return;
                }

                try {
                    // ---------------------------------------------------------
                    // 회원가입 API 호출
                    // ---------------------------------------------------------
                    const response = await fetch('/api/signup', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            username,
                            password,
                            name,
                            gender,
                            birthdate,
                            phone,
                            address,
                            address_detail: addressDetail
                        })
                    });

                    const data = await response.json();

                    if (data.success) {
                        showToast('회원가입 완료! 🎉');
                        setTimeout(() => {
                            window.location.href = data.redirect || '/login';
                        }, 1000);
                    } else {
                        showToast(data.message || '회원가입에 실패했습니다');
                    }
                } catch (error) {
                    console.error('Signup error:', error);
                    showToast('회원가입 중 오류가 발생했습니다');
                }
            });
        }
    }

    // =============================================================
    // 아이디 중복 확인
    // =============================================================

    // 전역 변수: 아이디 중복확인 여부
    window.usernameChecked = false;

    /**
     * 아이디 중복확인 버튼 이벤트 핸들러
     */
    function initUsernameCheck() {
        const checkBtn = document.getElementById('check-username-btn');
        const usernameInput = document.getElementById('username');
        const messageSpan = document.getElementById('username-message');

        if (checkBtn && usernameInput) {
            // 아이디 변경 시 중복확인 초기화
            usernameInput.addEventListener('input', function () {
                window.usernameChecked = false;
                if (messageSpan) {
                    messageSpan.textContent = '';
                    messageSpan.className = 'field-message';
                }
            });

            checkBtn.addEventListener('click', async function () {
                const username = usernameInput.value.trim();

                if (!username) {
                    showToast('아이디를 입력해주세요');
                    return;
                }

                if (username.length < 4) {
                    showToast('아이디는 4자 이상이어야 합니다');
                    return;
                }

                try {
                    // ---------------------------------------------------------
                    // 아이디 중복확인 API 호출
                    // ---------------------------------------------------------
                    const response = await fetch('/api/check-username', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ username })
                    });

                    const data = await response.json();

                    if (messageSpan) {
                        messageSpan.textContent = data.message;
                        if (data.available) {
                            messageSpan.className = 'field-message success';
                            window.usernameChecked = true;
                        } else {
                            messageSpan.className = 'field-message error';
                            window.usernameChecked = false;
                        }
                    }

                    showToast(data.message);
                } catch (error) {
                    console.error('Username check error:', error);
                    showToast('중복확인 중 오류가 발생했습니다');
                }
            });
        }
    }

    // =============================================================
    // 다음 주소 검색 API
    // =============================================================

    /**
     * 다음 우편번호 API를 이용한 주소 검색
     */
    function initAddressSearch() {
        const searchBtn = document.getElementById('search-address-btn');
        const addressInput = document.getElementById('address');
        const addressDetailInput = document.getElementById('address-detail');

        if (searchBtn && addressInput) {
            searchBtn.addEventListener('click', function () {
                // ---------------------------------------------------------
                // 다음 우편번호 API 호출
                // ---------------------------------------------------------
                new daum.Postcode({
                    oncomplete: function (data) {
                        // 주소 조합
                        let fullAddress = data.address;
                        let extraAddress = '';

                        // R: 도로명 주소, J: 지번 주소
                        if (data.addressType === 'R') {
                            if (data.bname !== '') {
                                extraAddress += data.bname;
                            }
                            if (data.buildingName !== '') {
                                extraAddress += (extraAddress !== '' ? ', ' + data.buildingName : data.buildingName);
                            }
                            fullAddress += (extraAddress !== '' ? ' (' + extraAddress + ')' : '');
                        }

                        // 주소 입력란에 설정
                        addressInput.value = fullAddress;

                        // 상세주소 입력란에 포커스
                        if (addressDetailInput) {
                            addressDetailInput.focus();
                        }
                    }
                }).open();
            });
        }
    }

    // =============================================================
    // 유틸리티 함수
    // =============================================================

    /**
     * 토스트 알림 표시
     * @param {string} message - 표시할 메시지
     */
    function showToast(message) {
        const existingToast = document.querySelector('.toast');
        if (existingToast) existingToast.remove();

        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            bottom: 40px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--text-primary);
            color: var(--bg-primary);
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 14px;
            z-index: 1000;
            animation: fadeIn 0.3s ease;
        `;
        document.body.appendChild(toast);

        setTimeout(() => toast.remove(), 2500);
    }
});
