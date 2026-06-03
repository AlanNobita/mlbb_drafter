const WS_URL = `ws://${window.location.hostname}:8765`;

let ws = null;
let reconnectAttempts = 0;
const MAX_RECONNECTAttempts = 5;

function connect() {
    ws = new WebSocket(WS_URL);
    
    ws.onopen = () => {
        console.log('Connected to server');
        document.getElementById('status').textContent = 'Connected';
        document.getElementById('status').className = 'status connected';
        reconnectAttempts = 0;
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateDashboard(data);
        if (data.strategy_flags) updateStrategyFlags(data.strategy_flags);
        if (data.counter_alerts) updateCounterAlerts(data.counter_alerts);
        if (data.role_lanes) updateRoleLanes(data.role_lanes);
    };
    
    ws.onclose = () => {
        console.log('Disconnected from server');
        document.getElementById('status').textContent = 'Disconnected';
        document.getElementById('status').className = 'status disconnected';
        
        if (reconnectAttempts < MAX_RECONNECTAttempts) {
            reconnectAttempts++;
            setTimeout(connect, 2000 * reconnectAttempts);
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

function updateDashboard(data) {
    // Update ally picks
    const allyPicks = document.getElementById('ally-picks');
    allyPicks.innerHTML = (data.ally_picks || [])
        .map(hero => `<span class="hero-tag ally">${hero}</span>`)
        .join('');
    
    // Update enemy picks
    const enemyPicks = document.getElementById('enemy-picks');
    enemyPicks.innerHTML = (data.enemy_picks || [])
        .map(hero => `<span class="hero-tag enemy">${hero}</span>`)
        .join('');
    
    // Update bans
    const bans = document.getElementById('bans');
    bans.innerHTML = (data.bans || [])
        .map(hero => `<span class="hero-tag ban">${hero}</span>`)
        .join('');
    
    // Update recommendations
    updateRecommendations('top-picks', data.recommendations?.top_picks || []);
    updateRecommendations('counter-picks', data.recommendations?.counter_picks || []);
    updateRecommendations('synergy-picks', data.recommendations?.synergy_picks || []);
}

function updateRecommendations(elementId, recommendations) {
    const element = document.getElementById(elementId);
    element.innerHTML = recommendations
        .map(rec => `
            <div class="rec-item">
                <span class="hero-name">${rec.hero}</span>
                <span class="win-rate">${(rec.win_rate * 100).toFixed(1)}%</span>
            </div>
        `)
        .join('');
}

function updateStrategyFlags(flags) {
    document.querySelectorAll('#strategy-flags .flag').forEach(function(el) {
        if (flags.includes(el.dataset.flag)) {
            el.classList.add('active');
        } else {
            el.classList.remove('active');
        }
    });
}

function updateCounterAlerts(alerts) {
    var container = document.getElementById('alerts-container');
    if (!container) return;
    if (!alerts || alerts.length === 0) {
        container.innerHTML = '<div class="alert-item info"><span class="alert-text">No counter alerts</span></div>';
        return;
    }
    container.innerHTML = alerts.map(function(a) {
        return '<div class="alert-item ' + (a.severity || 'warning') + '"><span class="alert-icon">&#9888;</span><span class="alert-text">' + a.message + '</span></div>';
    }).join('');
}

function updateRoleLanes(lanes) {
    if (!lanes) return;
    var mapping = {EXP: 'exp-heroes', Gold: 'gold-heroes', Mid: 'mid-heroes', Roam: 'roam-heroes', Jungle: 'jungle-heroes'};
    for (var lane in lanes) {
        var el = document.getElementById(mapping[lane]);
        if (el) el.textContent = lanes[lane].join(', ') || '-';
    }
}

// Connect on page load
connect();
