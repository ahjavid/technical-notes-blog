// Main JavaScript functionality for the technical blog

// Smooth scrolling for anchor links
document.addEventListener('DOMContentLoaded', function() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// Reading progress indicator
function createReadingProgress() {
    const progressBar = document.createElement('div');
    progressBar.id = 'reading-progress';
    progressBar.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 0%;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        z-index: 9999;
        transition: width 0.3s ease;
    `;
    
    document.body.appendChild(progressBar);
    
    window.addEventListener('scroll', function() {
        const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
        const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const scrolled = (winScroll / height) * 100;
        
        progressBar.style.width = scrolled + '%';
    });
}

// Initialize reading progress on article pages
if (document.querySelector('article')) {
    createReadingProgress();
}

// Copy code functionality
function addCodeCopyButtons() {
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach(codeBlock => {
        const pre = codeBlock.parentElement;
        pre.style.position = 'relative';
        
        const copyButton = document.createElement('button');
        copyButton.innerHTML = '<i class="fas fa-copy"></i>';
        copyButton.className = 'copy-code-btn';
        copyButton.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255, 255, 255, 0.2);
            border: none;
            color: white;
            padding: 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s ease;
        `;
        
        copyButton.addEventListener('click', async function() {
            try {
                await navigator.clipboard.writeText(codeBlock.textContent);
                copyButton.innerHTML = '<i class="fas fa-check"></i>';
                copyButton.style.background = 'rgba(46, 204, 113, 0.8)';
                
                setTimeout(() => {
                    copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                    copyButton.style.background = 'rgba(255, 255, 255, 0.2)';
                }, 2000);
            } catch (err) {
                console.error('Failed to copy code:', err);
            }
        });
        
        copyButton.addEventListener('mouseenter', function() {
            this.style.background = 'rgba(255, 255, 255, 0.3)';
        });
        
        copyButton.addEventListener('mouseleave', function() {
            this.style.background = 'rgba(255, 255, 255, 0.2)';
        });
        
        pre.appendChild(copyButton);
    });
}

// Initialize code copy buttons
document.addEventListener('DOMContentLoaded', addCodeCopyButtons);

// Table of contents generator for long posts
function generateTableOfContents() {
    const headings = document.querySelectorAll('article h2, article h3');
    if (headings.length < 3) return; // Only generate TOC for longer posts
    
    const toc = document.createElement('div');
    toc.className = 'table-of-contents';
    toc.innerHTML = '<h3><i class="fas fa-list"></i> Table of Contents</h3>';
    
    const tocList = document.createElement('ul');
    
    headings.forEach((heading, index) => {
        const id = heading.id || `heading-${index}`;
        if (!heading.id) heading.id = id;
        
        const listItem = document.createElement('li');
        listItem.className = heading.tagName.toLowerCase();
        
        const link = document.createElement('a');
        link.href = `#${id}`;
        link.textContent = heading.textContent;
        
        listItem.appendChild(link);
        tocList.appendChild(listItem);
    });
    
    toc.appendChild(tocList);
    
    // Insert TOC after the first paragraph
    const firstParagraph = document.querySelector('article p');
    if (firstParagraph) {
        firstParagraph.parentNode.insertBefore(toc, firstParagraph.nextSibling);
    }
}

// Social sharing functionality
function addSocialSharing() {
    const article = document.querySelector('article');
    if (!article) return;
    
    const title = document.title;
    const url = window.location.href;
    const text = document.querySelector('meta[name="description"]')?.content || title;
    
    const socialContainer = document.createElement('div');
    socialContainer.className = 'social-sharing';
    socialContainer.innerHTML = `
        <h4><i class="fas fa-share-alt"></i> Share this post</h4>
        <div class="social-buttons">
            <a href="https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}" target="_blank" class="social-btn twitter">
                <i class="fab fa-twitter"></i> Twitter
            </a>
            <a href="https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(url)}" target="_blank" class="social-btn linkedin">
                <i class="fab fa-linkedin"></i> LinkedIn
            </a>
            <a href="https://news.ycombinator.com/submitlink?u=${encodeURIComponent(url)}&t=${encodeURIComponent(title)}" target="_blank" class="social-btn hackernews">
                <i class="fas fa-newspaper"></i> Hacker News
            </a>
        </div>
    `;
    
    // Add at the end of the article
    article.appendChild(socialContainer);
}

// Performance monitoring (Core Web Vitals)
function monitorPerformance() {
    if (!window.performance) return;
    
    // Measure and log Core Web Vitals
    new PerformanceObserver((entryList) => {
        for (const entry of entryList.getEntries()) {
            console.log(`${entry.name}: ${entry.value}`);
            
            // You can send this data to analytics service
            if (typeof gtag !== 'undefined') {
                gtag('event', entry.name, {
                    value: Math.round(entry.value),
                    custom_parameter: entry.value
                });
            }
        }
    }).observe({entryTypes: ['measure', 'paint', 'layout-shift']});
}

// Initialize performance monitoring
if (typeof PerformanceObserver !== 'undefined') {
    monitorPerformance();
}

// Search functionality (if implemented later)
function initializeSearch() {
    const searchInput = document.querySelector('#search-input');
    if (!searchInput) return;
    
    // Simple client-side search implementation
    searchInput.addEventListener('input', function(e) {
        const query = e.target.value.toLowerCase();
        const posts = document.querySelectorAll('.post-item');
        
        posts.forEach(post => {
            const title = post.querySelector('h3')?.textContent.toLowerCase() || '';
            const content = post.querySelector('p')?.textContent.toLowerCase() || '';
            
            if (title.includes(query) || content.includes(query)) {
                post.style.display = 'block';
            } else {
                post.style.display = 'none';
            }
        });
    });
}

// Initialize all functionality
document.addEventListener('DOMContentLoaded', function() {
    generateTableOfContents();
    addSocialSharing();
    initializeSearch();
});

// Export functions for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        createReadingProgress,
        addCodeCopyButtons,
        generateTableOfContents,
        addSocialSharing,
        monitorPerformance
    };
}
