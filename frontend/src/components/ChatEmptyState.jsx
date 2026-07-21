import { Bot, Plus } from "lucide-react";

export default function ChatEmptyState({ onCreateChat }) {
  return (
    <div className="chat-welcome">
      <div className="welcome-icon">
        <Bot size={28} aria-hidden="true" />
      </div>
      <h2>Bạn cần hỗ trợ gì hôm nay?</h2>
      <p>
        Tạo cuộc trò chuyện mới rồi hỏi AI. Agent sẽ tìm trong toàn bộ tài liệu local và dùng web khi thiếu context.
      </p>
      <button className="btn-primary" onClick={onCreateChat} type="button">
        <Plus size={16} aria-hidden="true" />
        Bắt đầu cuộc trò chuyện
      </button>
    </div>
  );
}
