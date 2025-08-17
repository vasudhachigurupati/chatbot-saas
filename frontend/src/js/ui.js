export function showModal(modalId) {
    document.querySelectorAll('.modal').forEach(modal => {
        modal.classList.remove('active');
    });
    
    const modal = document.getElementById(`${modalId}-modal`);
    if (modal) {
        modal.classList.add('active');
        const alertContainer = modal.querySelector('[id$="-alerts"]');
        if (alertContainer) {
            alertContainer.innerHTML = '';
        }
    }
}

export function closeModal(modalId) {
    const modal = document.getElementById(`${modalId}-modal`);
    if (modal) {
        modal.classList.remove('active');
    }
}

export function showAlert(containerId, message, type) {
    const alertContainer = document.getElementById(`${containerId}-alerts`);
    if (alertContainer) {
        alertContainer.innerHTML = `
            <div class="alert alert-${type}">
                ${message}
            </div>
        `;
    }
}

export function showCustomDialog(title, message, isConfirm = false) {
    return new Promise((resolve) => {
        const dialogModal = document.getElementById('custom-dialog-modal');
        const dialogTitle = document.getElementById('dialog-title');
        const dialogMessage = document.getElementById('dialog-message');
        const dialogOk = document.getElementById('dialog-ok');
        const dialogCancel = document.getElementById('dialog-cancel');

        dialogTitle.textContent = title;
        dialogMessage.textContent = message;
        dialogModal.classList.add('active');

        if (isConfirm) {
            dialogCancel.style.display = 'inline-block';
            dialogOk.textContent = 'Confirm';
        } else {
            dialogCancel.style.display = 'none';
            dialogOk.textContent = 'OK';
        }

        const handleOk = () => {
            dialogModal.classList.remove('active');
            dialogOk.removeEventListener('click', handleOk);
            dialogCancel.removeEventListener('click', handleCancel);
            resolve(true);
        };

        const handleCancel = () => {
            dialogModal.classList.remove('active');
            dialogOk.removeEventListener('click', handleOk);
            dialogCancel.removeEventListener('click', handleCancel);
            resolve(false);
        };
        
        dialogOk.addEventListener('click', handleOk);
        if (isConfirm) {
            dialogCancel.addEventListener('click', handleCancel);
        }
    });
}

export function scrollToSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
    }
}

export function hideEmailVerificationPage() {
    document.getElementById('verification-section').classList.remove('active');
    document.getElementById('hero').classList.remove('hidden');
    document.getElementById('main-content').classList.remove('hidden');
}

export function showEmailVerificationPage() {
    document.getElementById('hero').classList.add('hidden');
    document.getElementById('main-content').classList.add('hidden');
    document.getElementById('dashboard').classList.remove('active');
    document.getElementById('verification-section').classList.add('active');
}

export function selectPlan(plan) {
    const registerPlan = document.getElementById('register-plan');
    if (registerPlan) {
        registerPlan.value = plan;
    }
    showModal('register');
}

// src/js/ui.js

// ... (other code)

// Keep this version in ONE place (prefer ui.js) and import it everywhere
export function showLandingPageKeepAuth() {
  document.getElementById('hero').classList.remove('hidden');
  document.getElementById('main-content').classList.remove('hidden');
  document.getElementById('verification-section').classList.remove('active');
  document.getElementById('dashboard').classList.remove('active');

  // keep logged-in header
  document.getElementById('auth-buttons')?.classList.add('hidden');
  document.getElementById('user-menu')?.classList.remove('hidden');

  document.getElementById('dashboard-nav-item')?.classList.remove('hidden');
}
