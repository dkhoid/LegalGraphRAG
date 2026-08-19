function escapeHtml(unsafe) {
    if (!unsafe) return '';
    return unsafe
         .toString()
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}

document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const factInput = document.getElementById('factInput');
    const topKInput = document.getElementById('topK');

    const emptyState = document.getElementById('emptyState');
    const loadingState = document.getElementById('loadingState');
    const analysisResult = document.getElementById('analysisResult');

    const partyTabs = document.getElementById('partyTabs');
    const partyContentContainer = document.getElementById('partyContentContainer');
    const template = document.getElementById('partyContentTemplate');

    // API Key DOM
    const updateKeyBtn = document.getElementById('updateKeyBtn');
    const apiKeyInput = document.getElementById('apiKey');
    const providerSelect = document.getElementById('provider');

    if (updateKeyBtn) {
        const savedKey = localStorage.getItem('legal_api_key');
        const savedProvider = localStorage.getItem('legal_ai_provider');

        if (savedKey) apiKeyInput.value = savedKey;
        if (savedProvider && providerSelect) providerSelect.value = savedProvider;

        updateKeyBtn.addEventListener('click', () => {
            const newKey = apiKeyInput.value.trim();
            if (newKey) localStorage.setItem('legal_api_key', newKey);
            else localStorage.removeItem('legal_api_key');

            if (providerSelect) localStorage.setItem('legal_ai_provider', providerSelect.value);

            alert("Đã lưu cấu hình AI vào trình duyệt.");
        });
    }

    analyzeBtn.addEventListener('click', async () => {
        const fact = factInput.value.trim();
        const topK = parseInt(topKInput.value) || 5;

        if (fact.length < 30) {
            alert("Vui lòng nhập tình tiết vụ án chi tiết hơn (ít nhất 30 ký tự).");
            factInput.focus();
            return;
        }

        // Set Loading State
        analyzeBtn.disabled = true;
        emptyState.classList.add('hidden');
        analysisResult.classList.add('hidden');
        loadingState.classList.remove('hidden');

        try {
            const apiKey = apiKeyInput.value.trim() || localStorage.getItem('legal_api_key');
            const provider = (providerSelect ? providerSelect.value : null) || localStorage.getItem('legal_ai_provider') || 'openai';

            const response = await fetch('/analyze_civil', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    fact: fact,
                    top_k: topK,
                    api_key: apiKey || null,
                    provider: provider
                })
            });

            if (!response.ok) throw new Error(`Lỗi server: ${response.status}`);

            const data = await response.json();

            // Render the complex array result
            if (data.results && data.results.length > 0) {
                renderMultiPartyResult(data.results);
            } else {
                throw new Error("API không trả về kết quả hợp lệ.");
            }

            loadingState.classList.add('hidden');
            analysisResult.classList.remove('hidden');

            if (window.lucide) lucide.createIcons();

        } catch (error) {
            console.error('API Error:', error);
            alert('Đã xảy ra lỗi khi gọi API: ' + error.message);
            loadingState.classList.add('hidden');
            emptyState.classList.remove('hidden');
        } finally {
            analyzeBtn.disabled = false;
        }
    });

    function renderMultiPartyResult(results) {
        partyTabs.innerHTML = '';
        partyContentContainer.innerHTML = '';

        results.forEach((partyData, index) => {
            const partyId = `party-${index}`;
            const partyName = partyData.name || `Bên ${index + 1}`;

            // 1. Create Tab
            const tabBtn = document.createElement('button');
            tabBtn.className = `party-tab ${index === 0 ? 'active' : ''}`;
            tabBtn.textContent = partyName;
            tabBtn.dataset.target = partyId;
            partyTabs.appendChild(tabBtn);

            // 2. Create Content
            const contentNode = template.content.cloneNode(true);
            const contentDiv = contentNode.querySelector('.party-content');
            contentDiv.id = partyId;
            if (index === 0) contentDiv.classList.add('active');

            // --- Populate Data ---

            // Confidence
            const confBadge = contentDiv.querySelector('.confidence-badge');
            const confText = contentDiv.querySelector('.confidence-text');
            const confIcon = contentDiv.querySelector('i');

            let confOverall = 0;
            let confDict = partyData.confidence || {};

            if (typeof confDict === 'number') {
                confOverall = confDict;
                confDict = { overall: confOverall, grade: confOverall > 0.75 ? "HIGH" : "MEDIUM" };
            } else {
                confOverall = confDict.overall || 0;
            }

            const confPercent = Math.round(confOverall * 100);
            confText.innerHTML = `Độ tin cậy: ${confPercent}% <span style="font-size: 0.85em; opacity: 0.8;">(${confDict.grade || 'N/A'})</span>`;

            if (confPercent >= 75) {
                // Default success style
                confIcon.setAttribute('data-lucide', 'check-circle');
            } else if (confPercent >= 45) {
                confBadge.classList.add('warning');
                confIcon.setAttribute('data-lucide', 'alert-triangle');
            } else {
                confBadge.classList.add('danger');
                confIcon.setAttribute('data-lucide', 'alert-octagon');
            }

            const judgeRes = partyData.judge_result || {};

            // Dispute Type
            const disputeEl = contentDiv.querySelector('.dispute-type-content');
            disputeEl.innerHTML = `<strong>${escapeHtml(judgeRes.dispute_type || 'Không xác định')}</strong>`;

            // Laws
            const lawsEl = contentDiv.querySelector('.laws-content');
            const usedLaws = partyData.used_laws || [];
            if (usedLaws.length > 0) {
                usedLaws.forEach(law => {
                    const span = document.createElement('span');
                    span.className = 'law-tag';
                    // Show Law ID or description
                    span.textContent = law.id || 'Luật';
                    if(law.text) span.title = law.text;
                    lawsEl.appendChild(span);
                });
            } else {
                lawsEl.innerHTML = '<em>Không có điều luật nào được trích xuất.</em>';
            }

            // Facts (Evidence)
            const factsEl = contentDiv.querySelector('.facts-content');
            const usedFacts = partyData.used_facts || [];
            if (usedFacts.length > 0) {
                usedFacts.forEach(fact => {
                    const div = document.createElement('div');
                    div.className = 'evidence-card';
                    div.innerHTML = `<p>${escapeHtml(fact.fact || fact.text || '')}</p>
                                     <span class="evidence-source">Độ tương đồng: ${Math.round((fact.similarity || 0)*100)}%</span>`;
                    factsEl.appendChild(div);
                });
            } else {
                factsEl.innerHTML = '<em>Không có tình tiết căn cứ nào.</em>';
            }

            // Resolution
            const resEl = contentDiv.querySelector('.resolution-content');
            const resText = escapeHtml(judgeRes.resolution_direction || 'Chưa có phân tích chi tiết.').replace(/\n/g, '<br>');
            resEl.innerHTML = `<p>${resText}</p>`;

            // Trace Log
            const traceList = contentDiv.querySelector('.trace-list');
            const trace = partyData.reasoning_trace || {};
            const traceLines = [
                `Tình tiết lấy từ Vector/Graph: ${trace.retrieved_facts_count || 0}`,
                `Luật tham chiếu lấy được: ${trace.retrieved_laws_count || 0}`,
                `Sử dụng Reranker: ${trace.use_reranker ? 'Có' : 'Không'}`,
                `Sử dụng Self-Consistent Judge: ${trace.use_self_consistent ? 'Có' : 'Không'}`,
                `Luật thực tế áp dụng sau chọn lọc: ${trace.used_laws_count || 0}`,
                `Tình tiết thực tế làm căn cứ: ${trace.used_facts_count || 0}`,
                `Mô hình Judge: ${trace.judge_chatbot || 'Mặc định'}`
            ];

            if (confDict.retrieval_quality !== undefined) {
                traceLines.push(`[Metrics] Chất lượng tìm kiếm: ${Math.round(confDict.retrieval_quality * 100)}%`);
                traceLines.push(`[Metrics] Mức độ áp dụng luật: ${Math.round(confDict.law_applicability * 100)}%`);
                traceLines.push(`[Metrics] Bắt buộc xem xét lại (Review Required): ${confDict.review_required ? 'CÓ' : 'KHÔNG'}`);
            }

            traceLines.forEach(line => {
                const li = document.createElement('li');
                li.innerHTML = `<i data-lucide="check"></i> ${line}`;
                traceList.appendChild(li);
            });

            partyContentContainer.appendChild(contentDiv);

            // 3. Tab Click Event
            tabBtn.addEventListener('click', () => {
                // Deactivate all
                document.querySelectorAll('.party-tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.party-content').forEach(c => c.classList.remove('active'));
                // Activate clicked
                tabBtn.classList.add('active');
                document.getElementById(partyId).classList.add('active');
            });
        });
    }
});
