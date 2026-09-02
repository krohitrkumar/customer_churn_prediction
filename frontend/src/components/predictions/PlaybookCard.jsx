import './PlaybookCard.css';

export default function PlaybookCard({ icon, category, action, index }) {
  return (
    <div
      className="playbook-card anim-fade-up"
      style={{ animationDelay: `${index * 60}ms` }}
    >
      <span className="playbook-icon" aria-hidden="true">{icon}</span>
      <div className="playbook-body">
        <span className="playbook-category">{category}</span>
        <p className="playbook-action">{action}</p>
      </div>
    </div>
  );
}
