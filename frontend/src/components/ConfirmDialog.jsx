export default function ConfirmDialog({ title, message, confirmLabel, loading, onCancel, onConfirm }) {
  return (
    <div className="overlay" role="dialog" aria-modal="true">
      <section className="confirm-dialog">
        <h2>{title}</h2>
        <p>{message}</p>
        <div className="confirm-actions">
          <button className="btn-ghost" disabled={loading} onClick={onCancel} type="button">
            Hủy
          </button>
          <button className="btn-danger" disabled={loading} onClick={onConfirm} type="button">
            {loading ? "Đang xóa..." : confirmLabel}
          </button>
        </div>
      </section>
    </div>
  );
}
