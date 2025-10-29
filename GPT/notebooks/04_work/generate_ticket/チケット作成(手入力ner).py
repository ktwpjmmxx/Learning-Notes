# ===== チケット管理システム（対応日・状態をタイトル上に配置、色付き＆整列修正版） =====

import ipywidgets as widgets
from IPython.display import display, clear_output

# ===== グローバル変数 =====
tickets = []
ticket_counter = 1

# ===== 入力欄 =====
date_input_top = widgets.Text(
    description='対応日:',
    placeholder='YYYY-MM-DD',
    layout=widgets.Layout(width='190px', margin='0 10px 0 -355px')
)

status_input_top = widgets.Dropdown(
    options=['未処理', '処理中', '完了'],
    value='未処理',
    description='状態:',
    layout=widgets.Layout(width='170px', margin='0 10px 0 -50px')
)

date_status_row = widgets.HBox(
    [date_input_top, status_input_top],
    layout=widgets.Layout(justify_content='center', align_items='center', margin='0 0 10px 0', gap='10px')
)

title_input = widgets.Text(description='タイトル:', layout=widgets.Layout(width='400px', margin='-5px 0 3px 2px'))
requester_input = widgets.Text(description='依頼者:', layout=widgets.Layout(width='400px'))
assignee_input = widgets.Text(description='担当者:', layout=widgets.Layout(width='400px'))
content_input = widgets.Textarea(description='内容:', layout=widgets.Layout(width='400px', height='100px'))

# ===== ボタンと検索欄 =====
create_button = widgets.Button(description='チケット作成', layout=widgets.Layout(width='120px', height='30px'), button_style='success')
search_input = widgets.Text(placeholder='キーワード検索', layout=widgets.Layout(width='250px', height='30px'))
search_button = widgets.Button(description='検索', layout=widgets.Layout(width='70px', height='30px'), button_style='info')

date_search_input = widgets.Text(placeholder='日付検索 (YYYY-MM-DD)', layout=widgets.Layout(width='200px', height='30px'))
date_search_button = widgets.Button(description='検索', layout=widgets.Layout(width='70px', height='30px'), button_style='primary')

# ===== 出力エリア =====
output_area = widgets.VBox([])

# ===== チケット作成関数 =====
def create_ticket(b):
    global ticket_counter
    ticket = {
        'id': ticket_counter,
        'title': title_input.value.strip(),
        'requester': requester_input.value.strip(),
        'assignee': assignee_input.value.strip(),
        'due': date_input_top.value.strip(),
        'status': status_input_top.value,
        'content': content_input.value.strip()
    }
    tickets.append(ticket)
    ticket_counter += 1
    display_tickets()
    clear_inputs()

def clear_inputs():
    title_input.value = ''
    requester_input.value = ''
    assignee_input.value = ''
    date_input_top.value = ''
    status_input_top.value = '未処理'
    content_input.value = ''

# ===== 状態変更関数 =====
def update_status(change, ticket_id):
    for t in tickets:
        if t['id'] == ticket_id:
            t['status'] = change['new']
            break
    display_tickets()

# ===== チケット表示関数 =====
def display_tickets(filtered=None):
    output_area.children = []
    display_list = filtered if filtered is not None else tickets
    ticket_widgets = []

    for t in display_list:
        # 状態による色分け
        color = '#FF4C4C' if t['status'] == '未処理' else '#FFD700' if t['status'] == '処理中' else '#1E90FF'

        # 情報部分
        info_html = widgets.HTML(
            f"""
            <div style="
                width: 430px;
                padding: 10px;
                border-radius: 6px;
                background-color: {color};
                box-sizing: border-box;
            ">
                <b>ID:</b> {t['id']} &nbsp;&nbsp;
                <b>対応日:</b> {t['due']}<br>
                <b>依頼者:</b> {t['requester']} &nbsp;&nbsp;
                <b>担当者:</b> {t['assignee']}<br>
                <b>タイトル:</b> {t['title']}<br>
                <b>内容:</b> {t['content']}
            </div>
            """
        )

        # 状態プルダウン
        status_dropdown = widgets.Dropdown(
            options=['未処理', '処理中', '完了'],
            value=t['status'],
            layout=widgets.Layout(width='120px', margin='5px 0 0 0')
        )
        status_dropdown.observe(lambda change, ticket_id=t['id']: update_status(change, ticket_id), names='value')

        # VBoxで縦並びにすることで枠内に収める
        card = widgets.VBox(
            [info_html, status_dropdown],
            layout=widgets.Layout(margin='5px 0', gap='5px')
        )

        ticket_widgets.append(card)

    output_area.children = ticket_widgets
# ===== 検索関数 =====
def search_tickets(b=None):
    keyword = search_input.value.strip()
    if keyword:
        filtered = [t for t in tickets if keyword in t['title'] or keyword in t['content'] or keyword in t['requester'] or keyword in t['assignee']]
        display_tickets(filtered)
    else:
        display_tickets()

def search_by_date(b):
    date = date_search_input.value.strip()
    if date:
        filtered = [t for t in tickets if t['due'] == date]
        display_tickets(filtered)
    else:
        display_tickets()

search_input.on_submit(search_tickets)

# ===== ボタンアクション =====
create_button.on_click(create_ticket)
search_button.on_click(search_tickets)
date_search_button.on_click(search_by_date)

# ===== レイアウト =====
input_box = widgets.VBox([
    date_status_row,
    title_input,
    requester_input,
    assignee_input,
    content_input
], layout=widgets.Layout(margin='0 0 10px 0'))

button_row = widgets.HBox([
    create_button,
    search_input,
    search_button,
    widgets.HTML("<div style='width:20px;'></div>"),  # スペース
    date_search_input,
    date_search_button
], layout=widgets.Layout(margin='5px 0', align_items='center', gap='10px'))

ui = widgets.VBox([
    widgets.HTML("<h3 style='margin:0 0 15px 0; line-height:1.2; font-size:20px;'>チケット管理システム</h3>"),
    input_box,
    button_row,
    output_area
], layout=widgets.Layout(width='700px'))

display(ui)
