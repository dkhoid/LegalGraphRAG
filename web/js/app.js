document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const factInput = document.getElementById('factInput');
    const topKInput = document.getElementById('topK');

    const emptyState = document.getElementById('emptyState');
    const loadingState = document.getElementById('loadingState');
    const analysisResult = document.getElementById('analysisResult');
    const referencesSection = document.getElementById('referencesSection');

    // Kết quả DOM
    const resDisputeType = document.getElementById('resDisputeType');
    const resLaws = document.getElementById('resLaws');
    const resResolution = document.getElementById('resResolution');
    const retrievedLawsList = document.getElementById('retrievedLawsList');
    const retrievedFactsList = document.getElementById('retrievedFactsList');

    // API Key DOM
    const updateKeyBtn = document.getElementById('updateKeyBtn');
    const apiKeyInput = document.getElementById('apiKey');
    const providerSelect = document.getElementById('provider');

    if (updateKeyBtn) {
        // Khôi phục key từ localStorage
        const savedKey = localStorage.getItem('openai_api_key');
        const savedProvider = localStorage.getItem('ai_provider');

        if (savedKey) {
            apiKeyInput.value = savedKey;
        }
        if (savedProvider && providerSelect) {
            providerSelect.value = savedProvider;
        }

        updateKeyBtn.addEventListener('click', async () => {
            const newKey = apiKeyInput.value.trim();
            if (!newKey) {
                alert("Vui lòng nhập API Key!");
                return;
            }

            updateKeyBtn.disabled = true;
            updateKeyBtn.textContent = "Đang lưu...";

            try {
                // Lưu key và provider vào localStorage thay vì gửi lên server
                localStorage.setItem('openai_api_key', newKey);
                if (providerSelect) {
                    localStorage.setItem('ai_provider', providerSelect.value);
                }
                alert("Đã lưu cấu hình AI vào trình duyệt (localStorage).");
            } catch (err) {
                alert("Lỗi lưu Key: " + err.message);
            } finally {
                updateKeyBtn.disabled = false;
                updateKeyBtn.textContent = "Cập nhật";
            }
        });
    }

    analyzeBtn.addEventListener('click', async () => {
        const fact = factInput.value.trim();
        const topK = parseInt(topKInput.value) || 5;

        if (!fact) {
            alert('Vui lòng nhập tình tiết vụ án!');
            factInput.focus();
            return;
        }

        if (fact.length < 50) {
            alert("Vui lòng miêu tả tình huống của bạn chi tiết hơn (ít nhất 15 từ) để hệ thống có thể tìm kiếm dữ liệu pháp lý chính xác nhất. Ví dụ: 'Tôi cho một người vay 500 triệu đồng có viết giấy tay, đã quá hạn...'");
            factInput.focus();
            return;
        }

        // Cập nhật UI sang trạng thái Loading
        analyzeBtn.disabled = true;
        emptyState.classList.add('hidden');
        analysisResult.classList.add('hidden');
        referencesSection.classList.add('hidden');
        loadingState.classList.remove('hidden');

        try {
            // Lấy API key và provider từ input hoặc localStorage
            const apiKey = apiKeyInput.value.trim() || localStorage.getItem('openai_api_key');
            const provider = (providerSelect ? providerSelect.value : null) || localStorage.getItem('ai_provider') || 'openai';

            // Gọi API
            const response = await fetch('/analyze_civil', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    fact: fact,
                    top_k: topK,
                    api_key: apiKey || null,
                    provider: provider
                })
            });

            if (!response.ok) {
                throw new Error(`Lỗi server: ${response.status}`);
            }

            const data = await response.json();

            // Xử lý dữ liệu trả về và render
            renderResult(data);

            // Re-initialize icons for newly added DOM elements if necessary
            if (window.lucide) {
                lucide.createIcons();
            }

        } catch (error) {
            console.error('API Error:', error);
            alert('Đã xảy ra lỗi khi gọi API: ' + error.message);

            // Trả lại state ban đầu
            loadingState.classList.add('hidden');
            emptyState.classList.remove('hidden');
        } finally {
            analyzeBtn.disabled = false;
        }
    });

    function renderResult(data) {
        const { analysis_result, retrieved_laws, retrieved_facts } = data;

        // 1. Render Analysis
        resDisputeType.innerHTML = `<strong>${escapeHtml(analysis_result.dispute_type || 'Không xác định')}</strong>`;

        // Render Laws Tags
        resLaws.innerHTML = '';
        const applicableLaws = analysis_result.applicable_laws || [];
        if (applicableLaws.length > 0) {
            applicableLaws.forEach(law => {
                const tag = document.createElement('span');
                tag.className = 'law-tag';
                tag.textContent = law;
                resLaws.appendChild(tag);
            });
        } else {
            resLaws.innerHTML = '<em>Không có luật nào được chỉ định rõ.</em>';
        }

        // Render Resolution
        resResolution.innerHTML = `<p>${escapeHtml(analysis_result.resolution_direction || '').replace(/\n/g, '<br>')}</p>`;

        // 2. Render References (GraphDB Data) - FULL TEXT
        renderList(retrievedLawsList, retrieved_laws, (law) => `
            <div class="ref-title">Điều ${escapeHtml(law.entry || law.id || 'Unknown')}</div>
            <div class="ref-similarity">Độ tương đồng: ${(law.similarity * 100).toFixed(2)}%</div>
            <div class="ref-desc">${escapeHtml(law.description)}</div>
        `);

        renderList(retrievedFactsList, retrieved_facts, (fact) => `
            <div class="ref-title">Case ID: ${escapeHtml(fact.caseId || fact.id || 'Unknown')}</div>
            <div class="ref-similarity">Độ tương đồng: ${(fact.similarity * 100).toFixed(2)}%</div>
            <div class="ref-desc">${escapeHtml(fact.description)}</div>
        `);

        // Switch UI state
        loadingState.classList.add('hidden');
        analysisResult.classList.remove('hidden');
        referencesSection.classList.remove('hidden');
    }

    function renderList(container, items, templateFn) {
        container.innerHTML = '';
        if (!items || items.length === 0) {
            container.innerHTML = '<li><em>Không tìm thấy dữ liệu liên quan.</em></li>';
            return;
        }

        items.forEach(item => {
            const li = document.createElement('li');
            li.innerHTML = templateFn(item);
            container.appendChild(li);
        });
    }

    function escapeHtml(unsafe) {
        if (!unsafe) return '';
        return String(unsafe)
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }
});
