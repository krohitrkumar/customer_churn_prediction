export default function Skeleton({ width, height = '16px', borderRadius = 'var(--radius-md)', className = '', style = {} }) {
  return (
    <span className={`skeleton ${className}`} aria-hidden="true"
      style={{ display: 'block', width: width || '100%', height, borderRadius, ...style }} />
  );
}
export function SkeletonCard() {
  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <Skeleton height="12px" width="40%" />
      <Skeleton height="32px" width="60%" />
      <Skeleton height="10px" width="30%" />
    </div>
  );
}
export function SkeletonRow() {
  return (
    <tr aria-hidden="true">
      {Array.from({ length: 7 }).map((_, i) => (
        <td key={i} style={{ padding: '12px 16px' }}>
          <Skeleton height="12px" width={i === 0 ? '70%' : '50%'} />
        </td>
      ))}
    </tr>
  );
}
