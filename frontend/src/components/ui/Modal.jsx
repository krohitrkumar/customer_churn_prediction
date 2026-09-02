import * as Dialog from '@radix-ui/react-dialog';
import './Modal.css';

export default function Modal({ open, onOpenChange, title, description, children, size = 'md', footer }) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="modal-overlay" />
        <Dialog.Content className={`modal-content modal--${size}`} aria-describedby={description ? 'modal-desc' : undefined}>
          <div className="modal-header">
            <Dialog.Title className="modal-title">{title}</Dialog.Title>
            <Dialog.Close className="modal-close" aria-label="Close dialog">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M2 2l12 12M14 2L2 14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
            </Dialog.Close>
          </div>
          {description && <Dialog.Description id="modal-desc" className="modal-description">{description}</Dialog.Description>}
          <div className="modal-body">{children}</div>
          {footer && <div className="modal-footer">{footer}</div>}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
