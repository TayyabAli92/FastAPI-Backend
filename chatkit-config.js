// ChatKit Configuration for Book RAG Agent
const chatkitConfig = {
  backendUrl: process.env.BACKEND_URL || 'http://localhost:3001',
  chatEndpoint: '/chat',
  healthEndpoint: '/health',

  // UI Configuration
  title: 'Book RAG Agent',
  subtitle: 'Ask questions about robotics concepts from the book',

  // Features
  enableTextSelection: true,
  enableCitations: true,
  enableHistory: true,

  // Styling
  theme: {
    primaryColor: '#3b82f6', // Tailwind blue-500
    secondaryColor: '#f3f4f6', // Tailwind gray-100
    userBubbleColor: '#dbeafe', // Tailwind blue-100
    agentBubbleColor: '#f3f4f6', // Tailwind gray-100
  }
};

// Export for use in frontend
if (typeof module !== 'undefined' && module.exports) {
  module.exports = chatkitConfig;
} else if (typeof window !== 'undefined') {
  window.ChatKitConfig = chatkitConfig;
}