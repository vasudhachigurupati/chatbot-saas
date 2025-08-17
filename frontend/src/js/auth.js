import { showModal, closeModal, showAlert, showCustomDialog } from './ui.js';
import { loadDashboardData, showDashboard } from './dashboard.js';

// Function to show landing page
export function showLandingPage() {
    document.getElementById('hero').classList.remove('hidden');
    document.getElementById('main-content').classList.remove('hidden');
    document.getElementById('verification-section').classList.remove('active');
    document.getElementById('dashboard').classList.remove('active');
    document.getElementById('auth-buttons').classList.remove('hidden');
    document.getElementById('user-menu').classList.add('hidden');
}

// Function to handle login process
export async function login(email, password) {
    try {
        const formData = new FormData();
        formData.append('username', email);
        formData.append('password', password);

        const response = await fetch(`${window.API_BASE_URL}/auth/login`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            window.authToken = data.access_token;
            localStorage.setItem('authToken', window.authToken);
            
            closeModal('login');
            await verifyToken();
            showAlert('login', 'Login successful!', 'success');
        } else {
            showAlert('login', data.detail || 'Login failed', 'error');
        }
    } catch (error) {
        console.error('Login error:', error);
        showAlert('login', 'Network error. Please try again.', 'error');
    }
}

// Function to handle user registration
export async function register(email, company, password, plan) {
    try {
        const response = await fetch(`${window.API_BASE_URL}/auth/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                email: email,
                company_name: company,
                password: password,
                plan: plan
            })
        });

        const data = await response.json();

        if (response.ok) {
            window.resendEmailValue = email;
            closeModal('register');
            showModal('email-verification');
            showAlert('register', 'Account created! Please check your email.', 'success');
        } else {
            showAlert('register', data.detail || 'Registration failed', 'error');
        }
    } catch (error) {
        console.error('Registration error:', error);
        showAlert('register', 'Network error. Please try again.', 'error');
    }
}

// Function to verify email from token
export async function verifyEmail(token) {
    const iconEl = document.getElementById('verification-icon');
    const titleEl = document.getElementById('verification-title');
    const messageEl = document.getElementById('verification-message');
    const actionsEl = document.getElementById('verification-actions');

    if (!iconEl || !titleEl || !messageEl || !actionsEl) {
        console.error("Verification UI elements not found in DOM!");
        return;
    }

    try {
        const response = await fetch(`${window.API_BASE_URL}/auth/verify-email`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ token: token })
        });

        const data = await response.json();

        if (response.ok) {
            iconEl.className = 'verification-icon';
            iconEl.innerHTML = '<i class="fas fa-check"></i>';
            titleEl.textContent = 'Email Verified Successfully!';
            messageEl.textContent = 'Your email has been verified. You can now log in to your account.';
            actionsEl.classList.remove('hidden');
            window.history.replaceState({}, document.title, window.location.pathname);
        } else {
            iconEl.className = 'verification-icon error';
            iconEl.innerHTML = '<i class="fas fa-times"></i>';
            titleEl.textContent = 'Verification Failed';
            messageEl.textContent = data.detail || 'The verification link is invalid or has expired.';
            actionsEl.classList.remove('hidden');
        }
    } catch (error) {
        console.error('Email verification error:', error);
        iconEl.className = 'verification-icon error';
        iconEl.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
        titleEl.textContent = 'Network Error';
        messageEl.textContent = 'Unable to verify email. Please check your connection and try again.';
        actionsEl.classList.remove('hidden');
    }
}

// Function to handle forgot password
export async function forgotPassword(email) {
    try {
        const response = await fetch(`${window.API_BASE_URL}/auth/forgot-password`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: email })
        });

        const data = await response.json();
        showAlert('forgot-password', data.message, 'success');
    } catch (error) {
        console.error('Forgot password error:', error);
        showAlert('forgot-password', 'Network error. Please try again.', 'error');
    }
}

// Function to resend verification email
export async function resendVerification() {
    if (!window.resendEmailValue) {
        await showCustomDialog('Resend Error', 'Please register an email address first.');
        return;
    }

    try {
        const response = await fetch(`${window.API_BASE_URL}/auth/resend-verification`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: window.resendEmailValue })
        });

        const data = await response.json();
        
        if(response.ok) {
            await showCustomDialog('Resent!', data.message);
        } else {
            await showCustomDialog('Error', data.detail || 'Failed to resend verification email.', 'error');
        }
    } catch (error) {
        console.error('Resend verification error:', error);
        await showCustomDialog('Network Error', 'Network error. Please try again.', 'error');
    }
}

// Function to check if a token is valid
export async function verifyToken() {
    try {
        const response = await fetch(`${window.API_BASE_URL}/auth/me`, {
            headers: { 'Authorization': `Bearer ${window.authToken}` }
        });

        if (response.ok) {
            window.currentUser = await response.json();
            showDashboard();
            loadDashboardData();
        } else {
            localStorage.removeItem('authToken');
            window.authToken = null;
            showLandingPage();
        }
    } catch (error) {
        console.error('Token verification failed:', error);
        localStorage.removeItem('authToken');
        window.authToken = null;
        showLandingPage();
    }
}

// Add these functions to auth.js to handle password reset properly

// Function to handle password reset (when user clicks reset link in email)
export async function resetPassword(token, newPassword) {
    try {
        const response = await fetch(`${window.API_BASE_URL}/auth/reset-password`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                token: token,
                new_password: newPassword
            })
        });

        const data = await response.json();

        if (response.ok) {
            await showCustomDialog('Password Reset Successful', 'Your password has been reset successfully. You can now login with your new password.');
            // Redirect to login after successful reset
            window.location.href = '/'; // or showModal('login');
        } else {
            await showCustomDialog('Reset Failed', data.detail || 'The reset link is invalid or has expired.');
        }
    } catch (error) {
        console.error('Password reset error:', error);
        await showCustomDialog('Network Error', 'Unable to reset password. Please check your connection and try again.');
    }
}

// Function to check if current page is a password reset page
export function checkPasswordResetToken() {
    const urlParams = new URLSearchParams(window.location.search);
    const resetToken = urlParams.get('reset_token');
    const action = urlParams.get('action');

    if (resetToken || action === 'reset-password') {
        showPasswordResetPage();
        return resetToken;
    }
    return null;
}


// Function to show password reset form
// In src/js/auth.js, update your existing `showPasswordResetPage` function.
export function showPasswordResetPage() {
    // This function will handle hiding all other content.
    // The previous code had a bug where `hideAllSections` wasn't being called here.
    document.getElementById('hero').classList.add('hidden');
    document.getElementById('main-content').classList.add('hidden');
    document.getElementById('dashboard').classList.remove('active');
    document.getElementById('verification-section').classList.remove('active');
    
    // Dynamically create the reset section if it doesn't exist
    let resetSection = document.getElementById('password-reset-section');
    if (!resetSection) {
        resetSection = createPasswordResetSection();
        document.body.appendChild(resetSection);
    }
    // Make sure the reset section is visible
    resetSection.classList.add('active');
    
    // You may want to hide the main navigation as well for a clean reset experience
    const navbar = document.querySelector('.navbar');
    if (navbar) {
        navbar.classList.add('hidden');
    }
}

// Function to create password reset section HTML
function createPasswordResetSection() {
    const resetSection = document.createElement('section');
    resetSection.id = 'password-reset-section';
    resetSection.className = 'verification-section';
    resetSection.innerHTML = `
        <div class="verification-card">
            <div class="verification-icon">
                <i class="fas fa-key"></i>
            </div>
            <h2>Reset Your Password</h2>
            <p>Enter your new password below:</p>
            <form id="reset-password-form" style="margin-top: 2rem;">
                <div class="form-group">
                    <label class="form-label">New Password</label>
                    <input type="password" class="form-input" id="reset-new-password" required minlength="6">
                </div>
                <div class="form-group">
                    <label class="form-label">Confirm New Password</label>
                    <input type="password" class="form-input" id="reset-confirm-password" required minlength="6">
                </div>
                <button type="submit" class="btn btn-primary" style="width: 100%;">
                    <i class="fas fa-save"></i> Reset Password
                </button>
            </form>
            <div style="margin-top: 2rem;">
                <button class="btn btn-outline" id="reset-back-to-login">
                    <i class="fas fa-arrow-left"></i> Back to Login
                </button>
            </div>
        </div>
    `;
    return resetSection;
}

// Function to log out
export function logout() {
    localStorage.removeItem('authToken');
    window.authToken = null;
    window.currentUser = null;
    showLandingPage();
}
