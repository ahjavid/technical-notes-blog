// SEO and social media optimization
const seoConfig = {
  siteName: "Technical Notes & Research Blog",
  siteUrl: "https://ahjavid.github.io/technical-notes-blog",
  author: "Ahmad Javid",
  description: "In-depth technical analysis and research insights on machine learning, system performance, and optimization",
  keywords: ["machine learning", "performance optimization", "distributed training", "technical blog", "research"],
  twitterHandle: "@ahjavid", // Update with your actual Twitter handle if different
  
  // Default meta tags for posts
  defaultPostMeta: {
    type: "article",
    section: "Technology",
    publishedTime: null, // Set dynamically per post
    modifiedTime: null,  // Set dynamically per post
    author: "Technical Research Blog",
    tags: []
  }
};

// Analytics configuration (uncomment and configure as needed)
const analyticsConfig = {
  // googleAnalytics: {
  //   measurementId: "G-XXXXXXXXXX"
  // },
  // plausibleAnalytics: {
  //   domain: "ahjavid.github.io"
  // }
};

// Performance monitoring
const performanceConfig = {
  // Monitor Core Web Vitals
  enableWebVitals: true,
  
  // Image optimization settings
  lazyLoading: true,
  imageOptimization: {
    quality: 85,
    formats: ['webp', 'jpeg'],
    sizes: [320, 640, 960, 1280, 1920]
  }
};

// Export configuration
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    seoConfig,
    analyticsConfig,
    performanceConfig
  };
}
