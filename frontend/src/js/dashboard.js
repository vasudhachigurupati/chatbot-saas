import { showModal, closeModal, showAlert, showCustomDialog } from './ui.js';

export function showDashboard() {
    document.getElementById('hero').classList.add('hidden');
    document.getElementById('main-content').classList.add('hidden');
    document.getElementById('verification-section').classList.remove('active');
    document.getElementById('dashboard').classList.add('active');
    document.getElementById('auth-buttons').classList.add('hidden');
    document.getElementById('user-menu').classList.remove('hidden');
    
    // Show dashboard nav link
    const dashboardNavItem = document.getElementById('dashboard-nav-item');
    if (dashboardNavItem) {
        dashboardNavItem.classList.remove('hidden');
    }
}

/*export function showLandingPageKeepAuth() {
    document.getElementById('hero').classList.remove('hidden');
    document.getElementById('main-content').classList.remove('hidden');
    document.getElementById('verification-section').classList.remove('active');
    document.getElementById('dashboard').classList.remove('active');
    // Keep user menu visible, hide auth buttons
    // (auth state is preserved)
}

export function showLandingPage() {
    document.getElementById('hero').classList.remove('hidden');
    document.getElementById('main-content').classList.remove('hidden');
    document.getElementById('verification-section').classList.remove('active');
    document.getElementById('dashboard').classList.remove('active');
    document.getElementById('auth-buttons').classList.remove('hidden');
    document.getElementById('user-menu').classList.add('hidden');
    
    // Hide dashboard nav link
    const dashboardNavItem = document.getElementById('dashboard-nav-item');
    if (dashboardNavItem) {
        dashboardNavItem.classList.add('hidden');
    }
}

export function showDashboardSection(sectionName) {
    document.querySelectorAll('.dashboard-section').forEach(section => {
        section.classList.add('hidden');
    });
    
    document.getElementById(`${sectionName}-section`).classList.remove('hidden');
    
    document.querySelectorAll('.sidebar-link').forEach(link => {
        link.classList.remove('active');
    });
    
    const clickedLink = event.target.closest('.sidebar-link');
    if (clickedLink) {
        clickedLink.classList.add('active');
    }
}*/

export function showDashboardSection(sectionName, clickedEl) {
  document.querySelectorAll('.dashboard-section').forEach(s => s.classList.add('hidden'));
  document.getElementById(`${sectionName}-section`)?.classList.remove('hidden');

  document.querySelectorAll('.sidebar-link').forEach(l => l.classList.remove('active'));
  if (clickedEl) clickedEl.classList.add('active');

  document.getElementById('sidebar-chatbots-link')?.addEventListener('click', (e) => {
  e.preventDefault();
  showDashboardSection('chatbots', e.currentTarget);
    });

}


// Data loading and display logic
export async function loadDashboardData() {
    try {
        const userResponse = await fetch(`${window.API_BASE_URL}/auth/me`, {
            headers: {
                'Authorization': `Bearer ${window.authToken}`
            }
        });

        if (userResponse.ok) {
            const userData = await userResponse.json();
            window.currentUser = userData;
            
            document.getElementById('total-chatbots').textContent = userData.stats.chatbots;
            document.getElementById('monthly-conversations').textContent = userData.stats.monthly_conversations;
            document.getElementById('current-plan').textContent = userData.plan.charAt(0).toUpperCase() + userData.plan.slice(1);
            
            document.getElementById('settings-email').value = userData.email;
            document.getElementById('settings-company').value = userData.company_name;
            document.getElementById('settings-plan').value = userData.plan.charAt(0).toUpperCase() + userData.plan.slice(1);

            const maxPagesInput = document.getElementById('chatbot-max-pages');
            const maxPagesLabel = document.getElementById('max-pages-label');
            const maxPagesLimit = userData.plan_limits.max_pages_per_bot;

            if (maxPagesInput && maxPagesLabel) {
                maxPagesInput.max = maxPagesLimit === -1 ? 1000 : maxPagesLimit;
                maxPagesInput.value = Math.min(parseInt(maxPagesInput.value), maxPagesInput.max);
                maxPagesLabel.innerHTML = `Max Pages to Scrape <small>(Your plan limit: ${maxPagesLimit === -1 ? 'Unlimited' : maxPagesLimit})</small>`;
            }
        }

        await loadChatbots();
    } catch (error) {
        console.error('Error loading dashboard data:', error);
    }
}

export async function loadChatbots() {
    try {
        const response = await fetch(`${window.API_BASE_URL}/chatbots/list`, {
            headers: {
                'Authorization': `Bearer ${window.authToken}`
            }
        });

        if (response.ok) {
            const chatbots = await response.json();
            displayChatbots(chatbots);
            
            const totalConversations = chatbots.reduce((sum, bot) => sum + bot.total_conversations, 0);
            document.getElementById('total-conversations').textContent = totalConversations;
        }
    } catch (error) {
        console.error('Error loading chatbots:', error);
    }
}

function displayChatbots(chatbots) {
    const chatbotList = document.getElementById('chatbot-list');
    
    if (chatbots.length === 0) {
        chatbotList.innerHTML = `
            <div style="padding: 2rem; text-align: center; color: var(--gray);">
                <i class="fas fa-robot" style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.3;"></i>
                <p>No chatbots created yet. Create your first chatbot to get started!</p>
            </div>
        `;
        return;
    }

    chatbotList.innerHTML = chatbots.map(chatbot => `
        <div class="chatbot-item" data-chatbot-id="${chatbot.id}">
            <div class="chatbot-info">
                <h4>${chatbot.name}</h4>
                <p>${chatbot.website_url} • ${chatbot.page_count} pages • ${chatbot.total_conversations} conversations</p>
            </div>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span class="status-badge status-${chatbot.status}">
                    ${chatbot.status.charAt(0).toUpperCase() + chatbot.status.slice(1)}
                </span>
                <div class="chatbot-actions">
                    ${chatbot.status === 'active' ? `
                        <button class="btn btn-outline btn-small action-embed">
                            <i class="fas fa-code"></i> Embed
                        </button>
                        <button class="btn btn-primary btn-small action-analytics">
                            <i class="fas fa-chart-bar"></i> Analytics
                        </button>
                    ` : ''}
                    <button class="btn-delete btn-small action-delete">
                        <i class="fas fa-trash-alt"></i> Delete
                    </button>
                </div>
            </div>
        </div>
    `).join('');
}

// Functions to handle specific API calls from the dashboard
export async function createChatbot(name, websiteUrl, sitemapUrl, maxPages, brandVoice) {
    try {
        const maxPagesInput = document.getElementById('chatbot-max-pages');
        if (parseInt(maxPages) > parseInt(maxPagesInput.max)) {
            showAlert('create-chatbot', `Page limit exceeded. Your plan allows a maximum of ${maxPagesInput.max} pages.`, 'error');
            return;
        }

        const response = await fetch(`${window.API_BASE_URL}/chatbots/create`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${window.authToken}`
            },
            body: JSON.stringify({
                name: name,
                website_url: websiteUrl,
                sitemap_url: sitemapUrl || null,
                max_pages: parseInt(maxPages),
                brand_voice: brandVoice
            })
        });

        const data = await response.json();

        if (response.ok) {
            closeModal('create-chatbot');
            showAlert('create-chatbot', 'Chatbot creation started! We\'re scraping your website...', 'success');
            
            setTimeout(() => {
                loadChatbots();
            }, 2000);
        } else {
            showAlert('create-chatbot', data.detail || 'Failed to create chatbot', 'error');
        }
    } catch (error) {
        console.error('Create chatbot error:', error);
        showAlert('create-chatbot', 'Network error. Please try again.', 'error');
    }
}

export async function showEmbedCode(chatbotId) {
    try {
        const response = await fetch(`${window.API_BASE_URL}/chatbots/${chatbotId}/embed-code`, {
            headers: {
                'Authorization': `Bearer ${window.authToken}`
            }
        });

        if (response.ok) {
            const data = await response.json();
            document.getElementById('embed-code-content').textContent = data.embed_code;
            showModal('embed-code');
        } else {
            const data = await response.json();
            await showCustomDialog('Error', data.detail || 'Failed to get embed code');
        }
    } catch (error) {
        console.error('Embed code error:', error);
        await showCustomDialog('Error', 'Network error. Please try again.');
    }
}

export async function copyEmbedCode() {
    const embedCode = document.getElementById('embed-code-content').textContent;
    try {
        await navigator.clipboard.writeText(embedCode);
        showCustomDialog('Success', 'Embed code copied to clipboard!');
    } catch (err) {
        console.error('Failed to copy using clipboard API. Falling back to execCommand.', err);
        const textArea = document.createElement('textarea');
        textArea.value = embedCode;
        document.body.appendChild(textArea);
        textArea.select();
        try {
            document.execCommand('copy');
            showCustomDialog('Success', 'Embed code copied to clipboard!');
        } catch (execErr) {
            console.error('Fallback copy failed:', execErr);
            showCustomDialog('Error', 'Failed to copy code. Please select the code and copy it manually.');
        } finally {
            document.body.removeChild(textArea);
        }
    }
}

export async function deleteChatbot(chatbotId) {
    const shouldDelete = await showCustomDialog(
        'Delete Chatbot', 
        'Are you sure you want to permanently delete this chatbot and all its data?',
        true
    );

    if (!shouldDelete) return;

    try {
        const response = await fetch(`${window.API_BASE_URL}/chatbots/${chatbotId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${window.authToken}`
            }
        });
        
        const data = await response.json();
        if (response.ok) {
            await showCustomDialog('Success', data.message);
            loadChatbots();
        } else {
            await showCustomDialog('Error', data.detail || 'Failed to delete chatbot.', 'error');
        }
    } catch (error) {
        console.error('Error deleting chatbot:', error);
        await showCustomDialog('Network Error', 'Network error. Please try again.', 'error');
    }
}

export async function viewAnalytics(chatbotId) {
    showDashboardSection('analytics');
    
    try {
        const response = await fetch(`${window.API_BASE_URL}/analytics/${chatbotId}`, {
            headers: {
                'Authorization': `Bearer ${window.authToken}`
            }
        });

        if (response.ok) {
            const data = await response.json();
            displayAnalytics(data);
        } else {
            const errorData = await response.json();
            document.getElementById('analytics-content').innerHTML = `
                <div class="alert alert-error">
                    ${errorData.detail || 'Failed to load analytics'}
                </div>
            `;
        }
    } catch (error) {
        console.error('Analytics error:', error);
        document.getElementById('analytics-content').innerHTML = `
            <div class="alert alert-error">
                Network error. Please try again.
            </div>
        `;
    }
}

function displayAnalytics(data) {
    const analyticsContent = document.getElementById('analytics-content');
    
    analyticsContent.innerHTML = `
        <!-- Overall Stats -->
        <div class="stats-grid" style="margin-bottom: 2rem;">
            <div class="stat-card">
                <div class="stat-value">${data.overall.total_conversations}</div>
                <div class="stat-label">Total Conversations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${data.overall.unique_sessions}</div>
                <div class="stat-label">Unique Sessions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${data.overall.avg_response_time_ms}ms</div>
                <div class="stat-label">Avg Response Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${data.overall.unique_visitors}</div>
                <div class="stat-label">Unique Visitors</div>
            </div>
        </div>

        <!-- Daily Stats Chart -->
        <div class="chart-container">
            <h3 class="chart-title">Daily Conversations (Last ${data.period_days} days)</h3>
            <div id="daily-chart" style="height: 300px; display: flex; align-items: end; gap: 4px; padding: 20px;">
                ${data.daily_stats.map(stat => `
                    <div style="
                        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
                        height: ${Math.max(stat.total_conversations * 10, 10)}px;
                        min-width: 20px;
                        border-radius: 4px 4px 0 0;
                        position: relative;
                        flex: 1;
                        max-width: 40px;
                    " title="${stat.date}: ${stat.total_conversations} conversations">
                        <div style="
                            position: absolute;
                            bottom: -25px;
                            left: 50%;
                            transform: translateX(-50%) rotate(-45deg);
                            font-size: 10px;
                            color: var(--gray);
                            white-space: nowrap;
                        ">${new Date(stat.date).getDate()}</div>
                    </div>
                `).join('')}
            </div>
        </div>

        <!-- Popular Topics -->
        <div class="popular-topics">
            <h3 class="chart-title">Popular Topics</h3>
            ${data.popular_topics.length > 0 ? data.popular_topics.map(topic => `
                <div class="topic-item">
                    <span style="flex: 1;">${topic.topic}</span>
                    <span class="topic-frequency">${topic.frequency}</span>
                </div>
            `).join('') : '<p style="text-align: center; color: var(--gray);">No topics data available</p>'}
        </div>
    `;
}

// Add this function to dashboard.js to handle plan downgrade protection

// Add this new function to handle the plan change logic
export async function upgradePlan(plan) {
    const planLimits = {
        'free': 1,
        'starter': 3,
        'professional': 10,
        'enterprise': -1
    };

    const newPlanLimit = planLimits[plan];
    const currentChatbotCount = parseInt(document.getElementById('total-chatbots').textContent);

    // Check if the new plan has a lower limit than the current number of chatbots
    if (newPlanLimit !== -1 && currentChatbotCount > newPlanLimit) {
        await showCustomDialog(
            'Cannot Change Plan',
            `You have ${currentChatbotCount} chatbots. Please delete ${currentChatbotCount - newPlanLimit} chatbot(s) before changing to the ${plan.charAt(0).toUpperCase() + plan.slice(1)} plan, which allows only ${newPlanLimit}.`,
            false
        );
        return;
    }

    const shouldChange = await showCustomDialog(
        'Change Plan',
        `Are you sure you want to change to the ${plan.charAt(0).toUpperCase() + plan.slice(1)} plan?`,
        true
    );

    if (!shouldChange) return;

    try {
        const response = await fetch(`${window.API_BASE_URL}/subscription/upgrade`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${window.authToken}`
            },
            body: JSON.stringify({ plan: plan })
        });

        const data = await response.json();

        if (response.ok) {
            await showCustomDialog('Success', 'Subscription updated successfully!');
            loadDashboardData();
        } else {
            await showCustomDialog('Error', data.detail || 'Failed to update subscription');
        }
    } catch (error) {
        console.error('Plan change error:', error);
        await showCustomDialog('Network Error', 'Network error. Please try again.', 'error');
    }
}
