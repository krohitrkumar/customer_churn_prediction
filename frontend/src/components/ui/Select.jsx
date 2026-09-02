import * as RadixSelect from '@radix-ui/react-select';
import './Select.css';

export default function Select({ label, error, id, options = [], placeholder, value, onValueChange, disabled, required }) {
  const selectId = id || `select-${Math.random().toString(36).slice(2,7)}`;
  return (
    <div className="field">
      {label && <label className="field-label" htmlFor={selectId}>{label}{required && <span className="field-required">*</span>}</label>}
      <RadixSelect.Root value={value} onValueChange={onValueChange} disabled={disabled}>
        <RadixSelect.Trigger id={selectId} className={`select-trigger ${error ? 'select-trigger--error' : ''}`} aria-invalid={!!error}>
          <RadixSelect.Value placeholder={placeholder || 'Select...'} />
          <RadixSelect.Icon className="select-chevron">
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2 4l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
          </RadixSelect.Icon>
        </RadixSelect.Trigger>
        <RadixSelect.Portal>
          <RadixSelect.Content className="select-content" position="popper" sideOffset={4}>
            <RadixSelect.Viewport className="select-viewport">
              {options.map((opt) => (
                <RadixSelect.Item key={opt.value} value={opt.value} className="select-item">
                  <RadixSelect.ItemText>{opt.label}</RadixSelect.ItemText>
                  <RadixSelect.ItemIndicator className="select-check">
                    <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>
                  </RadixSelect.ItemIndicator>
                </RadixSelect.Item>
              ))}
            </RadixSelect.Viewport>
          </RadixSelect.Content>
        </RadixSelect.Portal>
      </RadixSelect.Root>
      {error && <p className="field-error" role="alert">{error}</p>}
    </div>
  );
}
