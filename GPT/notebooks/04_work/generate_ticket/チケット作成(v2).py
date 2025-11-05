import ipywidgets as widgets
from IPython.display import display

# ===== 定数定義 =====
STATUS_OPTIONS = ['未処理', '処理中', '完了']
STATUS_COLORS = {
    '未処理': 'linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%)',
    '処理中': 'linear-gradient(135deg, #ffd93d 0%, #ffb037 100%)',
    '完了': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'
}
STATUS_ICONS = {
    '未処理': '⏳',
    '処理中': '🔄',
    '完了': '✅'
}

# ===== グローバル変数 =====
tickets = []
ticket_counter = 1
error_output = widgets.Output()

# ===== UIコンポーネント作成 =====
def create_input_widgets():
    """入力ウィジェットを作成"""
    date_input = widgets.Text(
        description='📅 対応日:',
        placeholder='YYYY-MM-DD',
        layout=widgets.Layout(width='220px'),
        style={'description_width': '80px'}
    )
    
    status_input = widgets.Dropdown(
        options=STATUS_OPTIONS,
        value='未処理',
        description='📊 状態:',
        layout=widgets.Layout(width='180px'),
        style={'description_width': '60px'}
    )
    
    date_status_row = widgets.HBox(
        [date_input, status_input],
        layout=widgets.Layout(margin='0 0 10px 0', gap='15px')
    )
    
    title_input = widgets.Text(
        description='📝 タイトル:',
        layout=widgets.Layout(width='500px'),
        style={'description_width': '80px'}
    )
    
    requester_input = widgets.Text(
        description='👤 依頼者:',
        layout=widgets.Layout(width='500px'),
        style={'description_width': '80px'}
    )
    
    assignee_input = widgets.Text(
        description='👨‍💼 担当者:',
        layout=widgets.Layout(width='500px'),
        style={'description_width': '80px'}
    )
    
    content_input = widgets.Textarea(
        description='📄 内容:',
        layout=widgets.Layout(width='500px', height='100px'),
        style={'description_width': '80px'}
    )
    
    return {
        'date': date_input,
        'status': status_input,
        'date_status_row': date_status_row,
        'title': title_input,
        'requester': requester_input,
        'assignee': assignee_input,
        'content': content_input
    }

def create_button_widgets():
    """ボタンウィジェットを作成"""
    create_btn = widgets.Button(
        description='チケット作成',
        layout=widgets.Layout(width='130px', height='35px'),
        button_style='success',
        tooltip='新しいチケットを作成'
    )
    
    search_input = widgets.Text(
        placeholder='🔍 キーワード検索',
        layout=widgets.Layout(width='200px', height='35px')
    )
    
    search_btn = widgets.Button(
        description='検索',
        layout=widgets.Layout(width='80px', height='35px'),
        button_style='info'
    )
    
    date_search_input = widgets.Text(
        placeholder='📅 日付検索 (YYYY-MM-DD)',
        layout=widgets.Layout(width='200px', height='35px')
    )
    
    date_search_btn = widgets.Button(
        description='日付検索',
        layout=widgets.Layout(width='100px', height='35px'),
        button_style='primary'
    )
    
    reset_btn = widgets.Button(
        description='リセット',
        layout=widgets.Layout(width='100px', height='35px'),
        button_style='warning'
    )
    
    return {
        'create': create_btn,
        'search_input': search_input,
        'search': search_btn,
        'date_search_input': date_search_input,
        'date_search': date_search_btn,
        'reset': reset_btn
    }

# ウィジェット初期化
inputs = create_input_widgets()
buttons = create_button_widgets()
output_area = widgets.VBox([])

# ===== エラー表示関数 =====
def show_error(message):
    """エラーメッセージをポップアップ風に表示"""
    error_html = widgets.HTML(
        f"""
        <div style='
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            font-family: "Segoe UI", sans-serif;
            font-size: 14px;
            animation: slideIn 0.3s ease-out;
        '>
            <strong>⚠️ エラー:</strong> {message}
        </div>
        <style>
            @keyframes slideIn {{
                from {{ opacity: 0; transform: translateY(-10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
        </style>
        """
    )
    error_output.clear_output()
    with error_output:
        display(error_html)
    
    # 3秒後に自動で消去
    import threading
    def clear_error():
        import time
        time.sleep(3)
        error_output.clear_output()
    
    thread = threading.Thread(target=clear_error)
    thread.daemon = True
    thread.start()

# ===== コア機能関数 =====
def create_ticket(b):
    """チケットを作成"""
    global ticket_counter
    
    # 入力値の取得とバリデーション
    title = inputs['title'].value.strip()
    if not title:
        show_error('タイトルは必須項目です。入力してください。')
        return
    
    ticket = {
        'id': ticket_counter,
        'title': title,
        'requester': inputs['requester'].value.strip(),
        'assignee': inputs['assignee'].value.strip(),
        'due': inputs['date'].value.strip(),
        'status': inputs['status'].value,
        'content': inputs['content'].value.strip()
    }
    
    tickets.append(ticket)
    ticket_counter += 1
    display_tickets()
    clear_inputs()
    error_output.clear_output()

def clear_inputs():
    """入力欄をクリア"""
    inputs['title'].value = ''
    inputs['requester'].value = ''
    inputs['assignee'].value = ''
    inputs['date'].value = ''
    inputs['status'].value = '未処理'
    inputs['content'].value = ''

def update_status(change, ticket_id):
    """チケットの状態を更新"""
    for ticket in tickets:
        if ticket['id'] == ticket_id:
            ticket['status'] = change['new']
            display_tickets()
            break

def delete_ticket(b, ticket_id):
    """チケットを削除"""
    global tickets
    tickets = [t for t in tickets if t['id'] != ticket_id]
    display_tickets()

def edit_ticket(ticket_id):
    """チケットを編集モードにする"""
    display_tickets(edit_mode_id=ticket_id)

def save_ticket(ticket_id, edit_widgets):
    """チケットの編集を保存"""
    # バリデーション
    title = edit_widgets['title'].value.strip()
    if not title:
        show_error('タイトルは必須項目です。入力してください。')
        return
    
    for ticket in tickets:
        if ticket['id'] == ticket_id:
            ticket['title'] = title
            ticket['requester'] = edit_widgets['requester'].value.strip()
            ticket['assignee'] = edit_widgets['assignee'].value.strip()
            ticket['due'] = edit_widgets['due'].value.strip()
            ticket['content'] = edit_widgets['content'].value.strip()
            break
    display_tickets()
    error_output.clear_output()

def cancel_edit():
    """編集をキャンセル"""
    display_tickets()

def create_ticket_card(ticket, is_edit_mode=False):
    """個別チケットカードを作成"""
    gradient = STATUS_COLORS[ticket['status']]
    icon = STATUS_ICONS[ticket['status']]
    ticket_id = ticket['id']
    
    if is_edit_mode:
        # 編集モード
        edit_title = widgets.Text(value=ticket['title'], layout=widgets.Layout(width='90%'))
        edit_requester = widgets.Text(value=ticket['requester'], layout=widgets.Layout(width='90%'))
        edit_assignee = widgets.Text(value=ticket['assignee'], layout=widgets.Layout(width='90%'))
        edit_due = widgets.Text(value=ticket['due'], layout=widgets.Layout(width='90%'))
        edit_content = widgets.Textarea(value=ticket['content'], layout=widgets.Layout(width='90%', height='80px'))
        
        edit_widgets = {
            'title': edit_title,
            'requester': edit_requester,
            'assignee': edit_assignee,
            'due': edit_due,
            'content': edit_content
        }
        
        save_btn = widgets.Button(description='💾 保存', button_style='success', layout=widgets.Layout(width='100px'))
        cancel_btn = widgets.Button(description='❌ キャンセル', button_style='danger', layout=widgets.Layout(width='110px'))
        
        save_btn.on_click(lambda b: save_ticket(ticket_id, edit_widgets))
        cancel_btn.on_click(lambda b: cancel_edit())
        
        card_html = widgets.HTML(
            f"""
            <style>
                .edit-card-{ticket_id} {{
                    background: {gradient};
                    border-radius: 12px;
                    padding: 16px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    color: white;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }}
                .edit-label-{ticket_id} {{
                    font-weight: 600;
                    margin: 8px 0 4px 0;
                    opacity: 0.9;
                    color: #333;
                }}
            </style>
            <div class="edit-card-{ticket_id}">
                <h3 style="margin-top:0;">{icon} チケット編集 (ID: {ticket_id})</h3>
                <div class="edit-label-{ticket_id}">📝 タイトル:</div>
            </div>
            """
        )
        
        return widgets.VBox([
            card_html,
            edit_title,
            widgets.HTML(f'<div class="edit-label-{ticket_id}" style="color:#333; margin-left:16px; font-weight:600;">👤 依頼者:</div>'),
            edit_requester,
            widgets.HTML(f'<div class="edit-label-{ticket_id}" style="color:#333; margin-left:16px; font-weight:600;">👨‍💼 担当者:</div>'),
            edit_assignee,
            widgets.HTML(f'<div class="edit-label-{ticket_id}" style="color:#333; margin-left:16px; font-weight:600;">📅 対応日:</div>'),
            edit_due,
            widgets.HTML(f'<div class="edit-label-{ticket_id}" style="color:#333; margin-left:16px; font-weight:600;">📄 内容:</div>'),
            edit_content,
            widgets.HBox([save_btn, cancel_btn], layout=widgets.Layout(gap='10px', margin='10px 0 0 16px'))
        ], layout=widgets.Layout(margin='8px 0'))
    
    # 通常モード
    status_dropdown = widgets.Dropdown(
        options=STATUS_OPTIONS,
        value=ticket['status'],
        layout=widgets.Layout(width='140px')
    )
    
    status_dropdown.observe(
        lambda change, tid=ticket_id: update_status(change, tid),
        names='value'
    )
    
    edit_btn = widgets.Button(
        description='✏️ 編集',
        button_style='info',
        layout=widgets.Layout(width='90px', height='32px')
    )
    
    delete_btn = widgets.Button(
        description='🗑️ 削除',
        button_style='danger',
        layout=widgets.Layout(width='90px', height='32px')
    )
    
    edit_btn.on_click(lambda b: edit_ticket(ticket_id))
    delete_btn.on_click(lambda b: delete_ticket(b, ticket_id))
    
    card_html = widgets.HTML(
        f"""
        <style>
            .ticket-card-{ticket_id} {{
                background: {gradient};
                border-radius: 12px;
                padding: 16px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1), 0 1px 3px rgba(0,0,0,0.08);
                transition: all 0.3s ease;
                color: white;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            .ticket-card-{ticket_id}:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 12px rgba(0,0,0,0.15), 0 3px 6px rgba(0,0,0,0.1);
            }}
            .ticket-header-{ticket_id} {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
                font-size: 14px;
                opacity: 0.95;
            }}
            .ticket-id-{ticket_id} {{
                font-size: 18px;
                font-weight: bold;
                background: rgba(255,255,255,0.2);
                padding: 4px 12px;
                border-radius: 20px;
            }}
            .ticket-content-{ticket_id} {{
                line-height: 1.6;
            }}
            .ticket-row-{ticket_id} {{
                margin: 6px 0;
                display: flex;
                align-items: flex-start;
            }}
            .ticket-label-{ticket_id} {{
                font-weight: 600;
                min-width: 70px;
                opacity: 0.9;
            }}
            .ticket-value-{ticket_id} {{
                flex: 1;
                word-wrap: break-word;
            }}
            .status-section-{ticket_id} {{
                margin-top: 12px;
                padding-top: 12px;
                border-top: 1px solid rgba(255,255,255,0.3);
                display: flex;
                align-items: center;
                gap: 10px;
            }}
        </style>
        <div class="ticket-card-{ticket_id}">
            <div class="ticket-header-{ticket_id}">
                <span class="ticket-id-{ticket_id}">{icon} ID: {ticket['id']}</span>
                <span>📅 {ticket['due'] or '未設定'}</span>
            </div>
            <div class="ticket-content-{ticket_id}">
                <div class="ticket-row-{ticket_id}">
                    <span class="ticket-label-{ticket_id}">📝 タイトル:</span>
                    <span class="ticket-value-{ticket_id}">{ticket['title']}</span>
                </div>
                <div class="ticket-row-{ticket_id}">
                    <span class="ticket-label-{ticket_id}">👤 依頼者:</span>
                    <span class="ticket-value-{ticket_id}">{ticket['requester'] or '未設定'}</span>
                </div>
                <div class="ticket-row-{ticket_id}">
                    <span class="ticket-label-{ticket_id}">👨‍💼 担当者:</span>
                    <span class="ticket-value-{ticket_id}">{ticket['assignee'] or '未設定'}</span>
                </div>
                <div class="ticket-row-{ticket_id}">
                    <span class="ticket-label-{ticket_id}">📄 内容:</span>
                    <span class="ticket-value-{ticket_id}">{ticket['content'] or '詳細なし'}</span>
                </div>
                <div class="status-section-{ticket_id}">
                    <span class="ticket-label-{ticket_id}">📊 状態:</span>
                </div>
            </div>
        </div>
        """
    )
    
    # 状態とボタンを横並びに
    bottom_row = widgets.HBox(
        [status_dropdown, edit_btn, delete_btn],
        layout=widgets.Layout(align_items='center', gap='10px', margin='8px 0 0 16px')
    )
    
    return widgets.VBox(
        [card_html, bottom_row],
        layout=widgets.Layout(margin='8px 0')
    )

def display_tickets(filtered=None, edit_mode_id=None):
    """チケット一覧を表示"""
    display_list = filtered if filtered is not None else tickets
    
    if not display_list:
        output_area.children = [
            widgets.HTML(
                """
                <div style='text-align:center; padding:40px; color:#999; font-size:16px;'>
                    📭 チケットがありません
                </div>
                """
            )
        ]
        return
    
    ticket_widgets = []
    for t in display_list:
        is_edit = (edit_mode_id == t['id'])
        ticket_widgets.append(create_ticket_card(t, is_edit_mode=is_edit))
    
    output_area.children = ticket_widgets

def search_tickets(b=None):
    """キーワード検索"""
    keyword = buttons['search_input'].value.strip().lower()
    if keyword:
        filtered = [
            t for t in tickets
            if keyword in t['title'].lower()
            or keyword in t['content'].lower()
            or keyword in t['requester'].lower()
            or keyword in t['assignee'].lower()
        ]
        display_tickets(filtered)
    else:
        display_tickets()

def search_by_date(b):
    """日付検索"""
    date = buttons['date_search_input'].value.strip()
    if date:
        filtered = [t for t in tickets if t['due'] == date]
        display_tickets(filtered)
    else:
        display_tickets()

def reset_search(b):
    """検索をリセット"""
    buttons['search_input'].value = ''
    buttons['date_search_input'].value = ''
    display_tickets()

# ===== イベントハンドラ登録 =====
buttons['create'].on_click(create_ticket)
buttons['search'].on_click(search_tickets)
buttons['date_search'].on_click(search_by_date)
buttons['reset'].on_click(reset_search)
buttons['search_input'].on_submit(search_tickets)

# ===== レイアウト構築 =====
input_box = widgets.VBox([
    inputs['date_status_row'],
    inputs['title'],
    inputs['requester'],
    inputs['assignee'],
    inputs['content']
], layout=widgets.Layout(margin='0 0 15px 0', gap='8px'))

button_row = widgets.HBox([
    buttons['create'],
    buttons['search_input'],
    buttons['search'],
    buttons['date_search_input'],
    buttons['date_search'],
    buttons['reset']
], layout=widgets.Layout(margin='10px 0', align_items='center', gap='10px'))

ui = widgets.VBox([
    widgets.HTML(
        """
        <h2 style='
            margin:0 0 20px 0;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
            text-align: center;
            font-family: "Segoe UI", sans-serif;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        '>
            🎫 チケット管理システム
        </h2>
        """
    ),
    error_output,
    input_box,
    button_row,
    widgets.HTML("<hr style='margin: 20px 0; border: none; border-top: 2px solid #eee;'>"),
    output_area
], layout=widgets.Layout(width='750px', padding='20px'))

# 初期表示
display(ui)
display_tickets()