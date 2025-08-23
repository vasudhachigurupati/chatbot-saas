 import { showAlert, showCustomDialog, showModal, closeModal, showDashboardSection } from './ui.js';

// Functions for API interactions
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
                window.loadChatbots();
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
            window.loadChatbots();
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


// frontend/src/js/api.js
export const API_BASE_URL =
  (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_BASE_URL)
    || (typeof window !== 'undefined' && window.API_BASE_URL)
    || 'http://127.0.0.1:8000';


async function displayAnalytics(data) {
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

export async function upgradePlan(plan) {
    const shouldUpgrade = await showCustomDialog(
        'Upgrade Plan',
        `Are you sure you want to upgrade to the ${plan.charAt(0).toUpperCase() + plan.slice(1)} plan?`,
        true
    );

    if (!shouldUpgrade) return;

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
            await showCustomDialog('Success', 'Subscription upgraded successfully!');
            window.loadDashboardData();
        } else {
            await showCustomDialog('Error', data.detail || 'Failed to upgrade subscription');
        }
    } catch (error) {
        console.error('Upgrade error:', error);
        await showCustomDialog('Network Error', 'Network error. Please try again.', 'error');
    }
}
