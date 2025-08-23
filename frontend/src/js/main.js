import { login, register, verifyToken, logout, verifyEmail, forgotPassword, resendVerification, resetPassword, checkPasswordResetToken, showPasswordResetPage } from './auth.js';
import { loadDashboardData, showDashboardSection, createChatbot, showEmbedCode, copyEmbedCode, deleteChatbot, viewAnalytics, upgradePlan } from './dashboard.js';
import { showModal, closeModal, showAlert, showCustomDialog, scrollToSection, hideEmailVerificationPage, showEmailVerificationPage, selectPlan, showLandingPage, showLandingPageKeepAuth } from './ui.js';
import { API_BASE_URL } from './api.js';


// Global state variables for the entire application
window.currentUser = null;
window.authToken = null;
window.resendEmailValue = null;
window.API_BASE_URL = 'http://127.0.0.1:8000'; // or 'http://localhost:8000'


// Function to go to login from verification page
export function goToLogin() {
    hideEmailVerificationPage();
    showModal('login');
}

// Function to go to home from verification page
export function goToHome() {
    hideEmailVerificationPage();
}

export function goToDashboard() {
    if (window.authToken && window.currentUser) {
        showDashboard();
        loadDashboardData();
    } else {
        showLandingPage();
    }
}


// In src/js/main.js, add the following function:
async function submitContactForm(name, email, subject, message) {
    try {
        const response = await fetch(`${API_BASE_URL}/contact/send`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                name: name,
                email: email,
                subject: subject,
                message: message,
            })
        });

        const data = await response.json();

        if (response.ok) {
            await showCustomDialog('Message Sent', 'Thank you for your message! We\'ll get back to you within 24 hours.');
            document.getElementById('contact-form').reset();
        } else {
            await showCustomDialog('Error', data.detail || 'Failed to send message. Please try again.');
        }
    } catch (error) {
        console.error('Contact form error:', error);
        await showCustomDialog('Error', 'Failed to send message. Network error. Please try again.');
    }
}

// In the `setupDynamicEventHandlers` function in `src/js/main.js`, add this event listener:
document.getElementById('contact-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = document.getElementById('contact-name').value;
    const email = document.getElementById('contact-email').value;
    const subject = document.getElementById('contact-subject').value;
    const message = document.getElementById('contact-message').value;
    await submitContactForm(name, email, subject, message);
});


window.showModal = showModal;
window.closeModal = closeModal;
window.showCustomDialog = showCustomDialog;
window.showDashboardSection = showDashboardSection;
window.showLandingPage = showLandingPage;
window.showLandingPageKeepAuth = showLandingPageKeepAuth;
window.goToDashboard = goToDashboard;
window.logout = logout;
window.selectPlan = selectPlan;
window.scrollToSection = scrollToSection;
window.showEmbedCode = showEmbedCode;
window.viewAnalytics = viewAnalytics;
window.deleteChatbot = deleteChatbot;
window.resendVerification = resendVerification;
window.copyEmbedCode = copyEmbedCode;
window.createChatbot = createChatbot;
window.upgradePlan = upgradePlan;
window.loadDashboardData = loadDashboardData;
window.login = login;
window.register = register;
window.forgotPassword = forgotPassword;
window.goToLogin = goToLogin;
window.goToHome = goToHome;
window.resetPassword = resetPassword;


function observeElements() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-fade-in-up');
            }
        });
    }, {
        threshold: 0.1
    });

    document.querySelectorAll('.feature-card').forEach(card => {
        observer.observe(card);
    });
}

function setupDynamicEventHandlers() {
    console.log('Setting up event handlers');

    document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        const href = link.getAttribute('href') || '';
        if (href.startsWith('#')) {
        e.preventDefault();                 // prevent navigation/reload
        const sectionId = href.substring(1);

        if (document.getElementById('dashboard').classList.contains('active')) {
            showLandingPageKeepAuth();
            setTimeout(() => scrollToSection(sectionId), 50);
        } else {
            scrollToSection(sectionId);
        }
        }
    });
    });
    // Add "Back to Dashboard" button to navigation when user is logged in
    const navMenu = document.querySelector('.nav-menu');
    if (navMenu && window.authToken) {
        let dashboardLink = document.getElementById('nav-dashboard-link');
        if (!dashboardLink) {
            const dashboardLi = document.createElement('li');
            dashboardLi.innerHTML = '<a href="#" id="nav-dashboard-link" class="nav-link">Dashboard</a>';
            navMenu.appendChild(dashboardLi);
            
            dashboardLink = document.getElementById('nav-dashboard-link');
            dashboardLink.addEventListener('click', (e) => {
                e.preventDefault();
                goToDashboard();
            });
        }
    }
    
    // Main Navigation Buttons
    const loginButton = document.getElementById('login-button');
    const registerButton = document.getElementById('register-button');
    
    if (loginButton) {
        loginButton.addEventListener('click', () => {
            showModal('login');
        });
    }
    
    if (registerButton) {
        registerButton.addEventListener('click', () => {
            showModal('register');
        });
    }

    document.getElementById('logout-button')?.addEventListener('click', () => logout());

    // Hero Section Buttons
    document.getElementById('hero-register-button')?.addEventListener('click', () => {
        showModal('register');
    });
    
    document.getElementById('hero-features-button')?.addEventListener('click', (e) => {
        e.preventDefault();
        scrollToSection('features');
    });

    // Pricing Buttons
    document.getElementById('plan-free-button')?.addEventListener('click', () => selectPlan('free'));
    document.getElementById('plan-starter-button')?.addEventListener('click', () => selectPlan('starter'));
    document.getElementById('plan-professional-button')?.addEventListener('click', () => selectPlan('professional'));
    document.getElementById('plan-enterprise-button')?.addEventListener('click', () => selectPlan('enterprise'));

    // Dashboard Quick Action Buttons
    document.getElementById('create-new-chatbot-button')?.addEventListener('click', () => showModal('create-chatbot'));
    document.getElementById('view-all-chatbots-button')?.addEventListener('click', (e) => { e.preventDefault(); showDashboardSection('chatbots'); });
    document.getElementById('view-analytics-button')?.addEventListener('click', (e) => { e.preventDefault(); showDashboardSection('analytics'); });
    document.getElementById('create-new-chatbot-button-2')?.addEventListener('click', () => showModal('create-chatbot'));

    // Verification Page Buttons
    document.getElementById('verification-login-button')?.addEventListener('click', () => goToLogin());
    document.getElementById('verification-home-button')?.addEventListener('click', () => goToHome());

    // Modal Close Buttons
    document.querySelectorAll('.modal-close').forEach(button => {
        button.addEventListener('click', (e) => {
            const modal = e.target.closest('.modal');
            if (modal) {
                const modalId = modal.id.replace('-modal', '');
                closeModal(modalId);
            }
        });
    });

    // Modal Link Toggles
    document.getElementById('forgot-password-link')?.addEventListener('click', (e) => { e.preventDefault(); showModal('forgot-password'); });
    document.getElementById('login-to-register-link')?.addEventListener('click', (e) => { e.preventDefault(); showModal('register'); });
    document.getElementById('register-to-login-link')?.addEventListener('click', (e) => { e.preventDefault(); showModal('login'); });
    document.getElementById('forgot-password-to-login-link')?.addEventListener('click', (e) => { e.preventDefault(); showModal('login'); });

    // Form Submissions
    document.getElementById('login-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = document.getElementById('login-email').value;
        const password = document.getElementById('login-password').value;
        await login(email, password);
    });
    
    document.getElementById('register-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = document.getElementById('register-email').value;
        const company = document.getElementById('register-company').value;
        const password = document.getElementById('register-password').value;
        const plan = document.getElementById('register-plan').value;
        await register(email, company, password, plan);
    });
    
    document.getElementById('forgot-password-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = document.getElementById('forgot-password-email').value;
        await forgotPassword(email);
    });
    
    document.getElementById('create-chatbot-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const name = document.getElementById('chatbot-name').value;
        const websiteUrl = document.getElementById('chatbot-website-url').value;
        const sitemapUrl = document.getElementById('chatbot-sitemap-url').value;
        const maxPages = document.getElementById('chatbot-max-pages').value;
        const brandVoice = document.getElementById('chatbot-brand-voice').value;
        await createChatbot(name, websiteUrl, sitemapUrl, maxPages, brandVoice);
    });
    
    document.getElementById('settings-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        await showCustomDialog('Success', 'Settings saved successfully!');
    });

    // Contact form - Updated to send to backend
    document.getElementById('contact-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const name = document.getElementById('contact-name').value;
        const email = document.getElementById('contact-email').value;
        const subject = document.getElementById('contact-subject').value;
        const message = document.getElementById('contact-message').value;
        await submitContactForm(name, email, subject, message);
    });
    
    document.getElementById('resend-verification-button')?.addEventListener('click', async () => {
        await resendVerification();
    });
    
    document.getElementById('copy-embed-code-button')?.addEventListener('click', () => copyEmbedCode());
    
    document.getElementById('upgrade-starter-button')?.addEventListener('click', () => upgradePlan('starter'));
    document.getElementById('upgrade-professional-button')?.addEventListener('click', () => upgradePlan('professional'));
    document.getElementById('upgrade-enterprise-button')?.addEventListener('click', () => upgradePlan('enterprise'));
    
    // Dashboard Sidebar Links
    document.getElementById('sidebar-overview-link')?.addEventListener('click', (e) => { e.preventDefault(); showDashboardSection('overview'); });
    document.getElementById('sidebar-chatbots-link')?.addEventListener('click', (e) => { e.preventDefault(); showDashboardSection('chatbots'); });
    document.getElementById('sidebar-analytics-link')?.addEventListener('click', (e) => { e.preventDefault(); showDashboardSection('analytics'); });
    document.getElementById('sidebar-settings-link')?.addEventListener('click', (e) => { e.preventDefault(); showDashboardSection('settings'); });
    document.getElementById('sidebar-billing-link')?.addEventListener('click', (e) => { e.preventDefault(); showDashboardSection('billing'); });

    // Password Reset Form Handler
    document.addEventListener('submit', async (e) => {
    if (e.target.id === 'reset-password-form') {
        e.preventDefault();
        const newPassword = document.getElementById('reset-new-password').value;
        const confirmPassword = document.getElementById('reset-confirm-password').value;
        
        if (newPassword !== confirmPassword) {
            await showCustomDialog('Error', 'Passwords do not match.');
            return;
        }
        
        const urlParams = new URLSearchParams(window.location.search);
        const resetToken = urlParams.get('reset_token');
        
        if (resetToken) {
            await resetPassword(resetToken, newPassword);
        }
    }
});


    // Back to login from reset page
    document.addEventListener('click', (e) => {
        if (e.target.id === 'reset-back-to-login') {
            window.location.href = '/';
        }
    });

    // Dynamic Button Handling using Event Delegation
    document.getElementById('chatbot-list')?.addEventListener('click', (e) => {
        const button = e.target.closest('button');
        if (!button) return;

        const chatbotItem = e.target.closest('.chatbot-item');
        const chatbotId = chatbotItem.dataset.chatbotId;

        if (button.classList.contains('action-embed')) {
            showEmbedCode(chatbotId);
        } else if (button.classList.contains('action-analytics')) {
            viewAnalytics(chatbotId);
        } else if (button.classList.contains('action-delete')) {
            deleteChatbot(chatbotId);
        }
    });
}

// In src/js/main.js, add this function to clear the UI state.
function hideAllSections() {
    document.getElementById('hero').classList.add('hidden');
    document.getElementById('main-content').classList.add('hidden');
    document.getElementById('dashboard').classList.remove('active');
    document.getElementById('verification-section').classList.remove('active');
    const resetSection = document.getElementById('password-reset-section');
    if (resetSection) {
        resetSection.classList.remove('active');
    }
}


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



function initializeApp() {
    console.log('Initializing app...');
    
    // Always hide the loading overlay and show the main container on startup
    document.getElementById('loading-overlay').classList.add('hidden');
    document.getElementById('app-container').classList.remove('hidden');
    
    const urlParams = new URLSearchParams(window.location.search);
    const verificationToken = urlParams.get('token');
    const resetToken = urlParams.get('reset_token');

    // Handle the most specific cases first
    if (resetToken) {
        console.log('Found reset_token. Showing password reset page.');
        showPasswordResetPage();
        // Since we're on a specific page, no other logic should run.
    } else if (verificationToken) {
        console.log('Found verification token. Showing email verification page.');
        showEmailVerificationPage();
        verifyEmail(verificationToken);
    } else {
        // This is the default path for normal Browse
        // Check for an existing session token
        window.authToken = localStorage.getItem('authToken');
        if (window.authToken) {
            console.log('Auth token found. Verifying and showing dashboard.');
            verifyToken();
        } else {
            console.log('No auth token found. Showing landing page.');
            showLandingPage();
        }
    }
    
    // Set up event handlers once after the initial page has been determined
    setupDynamicEventHandlers();
    observeElements();
}

document.addEventListener('DOMContentLoaded', initializeApp);