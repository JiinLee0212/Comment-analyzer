import sys
import io
sys.stdout.reconfigure(encoding='utf-8')

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import CountVectorizer
import re
import requests

import os
FONT_PATH = os.path.join(os.path.dirname(__file__), 'malgun.ttf')
if not os.path.exists(FONT_PATH):
    FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
from matplotlib import font_manager
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams['axes.unicode_minus'] = False

STOPWORDS = [
    '이', '그', '저', '것', '수', '등', '및', '를', '을', '이다',
    '하다', '있다', '되다', '않다', '없다', '같다', '보다', '주다',
    '좀', '더', '안', '못', '잘', '다', '또', '너무', '정말', '진짜',
    '그냥', '근데', '그리고', '하지만', '그래서', '영상', '댓글',
    '구독', '유튜브', 'ㅋㅋ', 'ㅠㅠ', 'ㄹㅇ', '기자', '뉴스', '기사'
]

# =============================================
# 유틸 함수
# =============================================
@st.cache_resource
def load_kiwi():
    return Kiwi()

def get_morphs(text, kiwi):
    if pd.isna(text): return []
    tokens = kiwi.tokenize(str(text))
    return [t.form for t in tokens
            if len(t.form) >= 2 and t.form not in STOPWORDS
            and (t.tag.startswith('NN') or t.tag.startswith('VA'))]

def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf

# =============================================
# YouTube 댓글 수집
# =============================================
def extract_video_id(url):
    patterns = [r'v=([^&]+)', r'youtu\.be/([^?]+)', r'youtube\.com/embed/([^?]+)']
    for p in patterns:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def collect_youtube_comments(api_key, video_id, max_comments=300):
    from googleapiclient.discovery import build
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    next_page_token = None
    while len(comments) < max_comments:
        req = youtube.commentThreads().list(
            part="snippet", videoId=video_id,
            maxResults=100, pageToken=next_page_token,
            textFormat="plainText", order="relevance"
        )
        res = req.execute()
        for item in res.get("items", []):
            s = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "댓글내용": s["textDisplay"],
                "좋아요수": s["likeCount"],
                "작성시간": s["publishedAt"],
                "작성자": s["authorDisplayName"]
            })
        next_page_token = res.get("nextPageToken")
        if not next_page_token: break
    return pd.DataFrame(comments[:max_comments])

# =============================================
# 네이버 뉴스 댓글 수집
# =============================================
def extract_naver_article_id(url):
    m = re.search(r'article/(\d+)/(\d+)', url)
    if m: return m.group(1), m.group(2)
    m = re.search(r'aid=(\d+).*oid=(\d+)', url)
    if m: return m.group(2), m.group(1)
    return None, None

def collect_naver_comments(url, max_comments=300):
    oid, aid = extract_naver_article_id(url)
    if not oid or not aid:
        raise ValueError("유효한 네이버 뉴스 URL을 입력해주세요.")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': url
    }

    comments = []
    page = 1

    while len(comments) < max_comments:
        api_url = (
            f"https://apis.naver.com/commentBox/cbox/web_neo_list_jsonp.json"
            f"?ticket=news&templateId=default&pool=cbox5"
            f"&lang=ko&country=KR&objectId=news{oid}%2C{aid}"
            f"&pageSize=100&page={page}&sort=FAVORITE"
        )
        res = requests.get(api_url, headers=headers)
        if res.status_code != 200: break

        text = res.text
        # JSONP 형태 파싱
        text = re.sub(r'^_callback\(', '', text).rstrip(');')
        import json
        data = json.loads(text)

        result = data.get('result', {})
        comment_list = result.get('commentList', [])
        if not comment_list: break

        for c in comment_list:
            if c.get('deleted', False): continue
            comments.append({
                "댓글내용": c.get('contents', ''),
                "좋아요수": c.get('sympathyCount', 0),
                "싫어요수": c.get('antipathyCount', 0),
                "작성시간": c.get('regTime', ''),
                "작성자": c.get('maskedUserId', '')
            })

        total = result.get('count', {}).get('comment', 0)
        if len(comments) >= total or len(comments) >= max_comments: break
        page += 1

    return pd.DataFrame(comments[:max_comments])

# =============================================
# 분석 탭 공통 함수
# =============================================
def show_keyword_tab(df, kiwi):
    st.subheader("키워드 분석")
    with st.spinner("키워드 분석 중..."):
        all_words = []
        for text in df['댓글내용']:
            all_words.extend(get_morphs(text, kiwi))
        counter = Counter(all_words)
        top_keywords = counter.most_common(20)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**상위 20개 키워드**")
        kw_df = pd.DataFrame(top_keywords, columns=['키워드', '빈도'])
        st.dataframe(kw_df, use_container_width=True)

    with col2:
        st.markdown("**키워드 막대그래프**")
        fig, ax = plt.subplots(figsize=(8, 6))
        words = [w for w, _ in top_keywords[:15]]
        counts = [c for _, c in top_keywords[:15]]
        ax.barh(words[::-1], counts[::-1], color='steelblue')
        ax.set_xlabel('언급 횟수')
        ax.set_title('상위 15개 키워드')
        plt.tight_layout()
        st.image(fig_to_image(fig))
        plt.close()

    st.markdown("**워드클라우드**")
    if counter:
        wc = WordCloud(font_path=FONT_PATH, width=800, height=400,
                       background_color='white', max_words=100, colormap='Blues')
        wc.generate_from_frequencies(dict(counter))
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout()
        st.image(fig_to_image(fig))
        plt.close()

def show_topic_tab(df, kiwi, num_topics):
    st.subheader("토픽 모델링 (BERTopic)")

    with st.expander("토픽 모델링이란? (클릭해서 펼치기)"):
        st.markdown("""
        ### 토픽 모델링이란?
        토픽 모델링은 대량의 텍스트에서 **숨겨진 주제(토픽)를 자동으로 찾아내는 기술**이에요.
        사람이 직접 댓글을 읽지 않아도, 컴퓨터가 비슷한 단어들끼리 묶어서 주요 주제를 파악해줘요.

        ### 어떤 모델을 사용하나요?
        이 앱은 **BERTopic** 모델을 사용해요.
        - 기존 LDA 모델보다 **문장의 의미를 더 잘 이해**해요
        - BERT 언어 모델이 문장 전체의 맥락을 파악해서 토픽을 구성해요
        - 토픽 수를 자동으로 결정하거나 원하는 수로 설정할 수 있어요

        ### 어떻게 해석하면 좋을까요?
        1. **토픽별 댓글 분포 그래프**: 어떤 주제가 가장 많이 언급됐는지 파악해요
        2. **핵심 키워드**: 각 토픽을 대표하는 단어들이에요. 키워드를 보고 주제를 직접 해석해보세요
           - 예: `최재훈 / 국대 / 탈락` → "선수 선발 논란" 주제로 해석 가능
        3. **좋아요 TOP 5 대표 댓글**: 실제 댓글을 보면서 토픽의 맥락을 확인하세요
           - 좋아요가 많은 댓글 = 많은 사람이 공감한 의견
        4. **주의사항**: 토픽 수가 너무 많으면 비슷한 주제가 쪼개질 수 있어요.
           댓글 수에 따라 3~6개가 적당해요
        """)

    with st.spinner("토픽 모델링 중... (시간이 걸릴 수 있어요)"):
        from bertopic import BERTopic

        docs_morphs = [' '.join(get_morphs(t, kiwi)) for t in df['댓글내용'].fillna('')]
        docs_morphs = [d if d.strip() else '기타' for d in docs_morphs]

        vectorizer = CountVectorizer(min_df=2, stop_words=STOPWORDS)
        topic_model = BERTopic(
            vectorizer_model=vectorizer,
            language='multilingual',
            min_topic_size=3,
            nr_topics=num_topics,
            calculate_probabilities=True,
            verbose=False
        )
        topics, probs = topic_model.fit_transform(docs_morphs)
        df = df.copy()
        df['토픽번호'] = topics

        topic_info = topic_model.get_topic_info()
        topic_info = topic_info[topic_info['Topic'] != -1].sort_values('Count', ascending=False)

        # 토픽 키워드 매핑
        topic_keyword_map = {}
        for _, row in topic_info.iterrows():
            t = row['Topic']
            words = topic_model.get_topic(t)
            topic_keyword_map[t] = ' / '.join([w for w, _ in words[:5]])

        df['토픽키워드'] = df['토픽번호'].apply(
            lambda x: topic_keyword_map.get(x, '미분류') if x != -1 else '미분류'
        )
        st.session_state['topic_df'] = df

    if len(topic_info) == 0:
        st.warning("댓글 수가 너무 적어 토픽 모델링이 어려워요.")
        return

    # 전체 분포 그래프
    st.markdown("### 토픽별 댓글 분포")
    topic_labels = []
    for _, row in topic_info.iterrows():
        t = row['Topic']
        words = topic_model.get_topic(t)
        keywords = ' / '.join([w for w, _ in words[:3]])
        topic_labels.append(f'토픽{t+1}: {keywords}')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(topic_labels[::-1], topic_info['Count'].values[::-1], color='steelblue')
    ax.set_xlabel('댓글 수')
    ax.set_title('토픽별 댓글 분포 (빈도순)')
    for i, v in enumerate(topic_info['Count'].values[::-1]):
        ax.text(v + 0.3, i, str(v), va='center')
    plt.tight_layout()
    st.image(fig_to_image(fig))
    plt.close()

    # 토픽별 상세
    st.markdown("### 토픽별 상세 분석")
    for rank, (_, row) in enumerate(topic_info.iterrows()):
        t = row['Topic']
        words = topic_model.get_topic(t)
        keywords = ' / '.join([w for w, _ in words[:5]])
        topic_df = df[df['토픽번호'] == t]
        count = row['Count']

        with st.expander(f"#{rank+1} 토픽 {t+1}: {keywords} ({count}개 댓글)"):

            # 키워드 점수 시각화
            st.markdown("**핵심 키워드 가중치**")
            kw_words = [w for w, _ in words[:8]]
            kw_scores = [s for _, s in words[:8]]
            fig, ax = plt.subplots(figsize=(8, 3))
            bars = ax.barh(kw_words[::-1], kw_scores[::-1], color='coral')
            ax.set_xlabel('가중치')
            ax.set_title(f'토픽 {t+1} 핵심 키워드')
            for bar, val in zip(bars, kw_scores[::-1]):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', fontsize=9)
            plt.tight_layout()
            st.image(fig_to_image(fig))
            plt.close()

            # 좋아요 TOP 5 대표 댓글
            st.markdown("**좋아요 TOP 5 대표 댓글**")
            top_comments = topic_df.nlargest(5, '좋아요수')
            for _, crow in top_comments.iterrows():
                likes = crow['좋아요수']
                content = str(crow['댓글내용'])[:120]
                st.info(f"👍 {likes}개 | {content}")

            # 전체 댓글 보기
            with st.popover(f"전체 댓글 보기 ({count}개)"):
                st.dataframe(
                    topic_df[['댓글내용', '좋아요수', '토픽키워드']].sort_values('좋아요수', ascending=False),
                    use_container_width=True
                )

def show_likes_tab(df):
    st.subheader("좋아요 가중치 분석")

    has_dislike = '싫어요수' in df.columns

    # 기본 통계
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 댓글 수", f"{len(df)}개")
    col2.metric("총 좋아요 수", f"{df['좋아요수'].sum():,}개")
    col3.metric("평균 좋아요", f"{df['좋아요수'].mean():.1f}개")
    col4.metric("최대 좋아요", f"{df['좋아요수'].max():,}개")

    # 좋아요 TOP 20
    st.markdown("### 좋아요 TOP 20 댓글")
    top_df = df.nlargest(20, '좋아요수')
    cols_show = ['댓글내용', '좋아요수']
    if has_dislike: cols_show.append('싫어요수')
    st.dataframe(top_df[cols_show], use_container_width=True)

    # 구간별 분석
    st.markdown("### 좋아요 구간별 분석")

    def get_tier(likes):
        if likes >= 100: return '핵심여론 (100+)'
        elif likes >= 10: return '공감여론 (10~99)'
        else: return '일반댓글 (~9)'

    df = df.copy()
    df['구간'] = df['좋아요수'].apply(get_tier)
    tier_order = ['핵심여론 (100+)', '공감여론 (10~99)', '일반댓글 (~9)']
    tier_colors = ['#FF6B6B', '#FFA94D', '#74C0FC']

    # 구간별 댓글 수
    col1, col2, col3 = st.columns(3)
    for col, tier, color in zip([col1, col2, col3], tier_order, tier_colors):
        tier_df = df[df['구간'] == tier]
        col.metric(tier, f"{len(tier_df)}개")

    # 구간별 키워드 비교
    st.markdown("**구간별 핵심 키워드 비교**")
    cols = st.columns(3)
    for col, tier, color in zip(cols, tier_order, tier_colors):
        with col:
            st.markdown(f"**{tier}**")
            tier_df = df[df['구간'] == tier]
            if len(tier_df) > 0:
                tier_words = []
                for text in tier_df['댓글내용']:
                    tier_words.extend(get_morphs(text, kiwi))
                tier_counter = Counter(tier_words).most_common(8)
                if tier_counter:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.barh([w for w, _ in tier_counter[::-1]],
                            [c for _, c in tier_counter[::-1]], color=color, alpha=0.8)
                    ax.set_title(f'{tier} 키워드')
                    plt.tight_layout()
                    st.image(fig_to_image(fig))
                    plt.close()

    # 네이버 전용: 싫어요 분석
    if has_dislike:
        st.markdown("### 논란 댓글 분석 (좋아요 - 싫어요)")
        df['논란지수'] = df['싫어요수'] / (df['좋아요수'] + df['싫어요수'] + 1)
        st.markdown("**가장 논란이 많은 댓글 TOP 10**")
        controversial = df.nlargest(10, '싫어요수')[['댓글내용', '좋아요수', '싫어요수', '논란지수']]
        st.dataframe(controversial, use_container_width=True)

# =============================================
# 메인 앱
# =============================================
st.set_page_config(page_title="댓글 분석기", page_icon="📊", layout="wide")

# 첫 화면: 소스 선택
if 'source' not in st.session_state:
    st.session_state.source = None

if st.session_state.source is None:
    st.title("📊 댓글 분석기")
    st.markdown("### 분석할 댓글 소스를 선택해주세요")
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='text-align:center; padding:40px; border:2px solid #FF0000; border-radius:15px;'>
            <h2>🎬 YouTube</h2>
            <p>YouTube 영상 댓글 분석</p>
            <p>API 키 필요</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("YouTube 선택", use_container_width=True, type="primary"):
            st.session_state.source = 'youtube'
            st.rerun()

    with col2:
        st.markdown("""
        <div style='text-align:center; padding:40px; border:2px solid #03C75A; border-radius:15px;'>
            <h2>📰 네이버 뉴스</h2>
            <p>네이버 뉴스 댓글 분석</p>
            <p>API 키 불필요</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("네이버 뉴스 선택", use_container_width=True, type="primary"):
            st.session_state.source = 'naver'
            st.rerun()

else:
    source = st.session_state.source
    source_name = "YouTube" if source == 'youtube' else "네이버 뉴스"
    icon = "🎬" if source == 'youtube' else "📰"

    st.title(f"{icon} {source_name} 댓글 분석기")

    if st.button("← 처음으로"):
        st.session_state.source = None
        st.rerun()

    with st.sidebar:
        st.header("설정")

        if source == 'youtube':
            api_key = st.text_input("YouTube API 키", type="password", placeholder="AIza...")
            with st.expander("API 키 발급 방법"):
                st.markdown("""
                1. [Google Cloud Console](https://console.cloud.google.com/) 접속
                2. 새 프로젝트 생성
                3. **API 및 서비스 → 라이브러리** 클릭
                4. `YouTube Data API v3` 검색 → **사용 설정**
                5. **사용자 인증 정보 → API 키 만들기**
                6. 생성된 키(`AIza...`) 복사해서 위에 입력

                > 무료 쿼터: 하루 약 1,000~2,000개 댓글 수집 가능
                """)
            url_input = st.text_input("YouTube 영상 URL", placeholder="https://www.youtube.com/watch?v=...")
        else:
            api_key = None
            url_input = st.text_input("네이버 뉴스 URL", placeholder="https://n.news.naver.com/article/...")

        max_comments = st.slider("최대 댓글 수집 수", 50, 2000, 200, 50)
        num_topics = st.slider("토픽 수", 2, 10, 5)
        analyze_btn = st.button("분석 시작!", type="primary", use_container_width=True)

    if analyze_btn:
        if not url_input:
            st.error("URL을 입력해주세요!")
            st.stop()
        if source == 'youtube' and not api_key:
            st.error("YouTube API 키를 입력해주세요!")
            st.stop()

        with st.spinner("댓글 수집 중..."):
            try:
                if source == 'youtube':
                    video_id = extract_video_id(url_input)
                    if not video_id:
                        st.error("유효한 YouTube URL을 입력해주세요!")
                        st.stop()
                    df = collect_youtube_comments(api_key, video_id, max_comments)
                else:
                    df = collect_naver_comments(url_input, max_comments)

                df['작성시간'] = pd.to_datetime(df['작성시간'], errors='coerce')
                st.success(f"댓글 {len(df)}개 수집 완료!")
            except Exception as e:
                st.error(f"댓글 수집 실패: {e}")
                st.stop()

        # 기본 통계
        col1, col2, col3 = st.columns(3)
        col1.metric("총 댓글 수", f"{len(df)}개")
        col2.metric("평균 좋아요", f"{df['좋아요수'].mean():.1f}개")
        valid_dates = df['작성시간'].dropna()
        if len(valid_dates) > 0:
            col3.metric("댓글 기간", f"{valid_dates.min().strftime('%m/%d')} ~ {valid_dates.max().strftime('%m/%d')}")

        kiwi = load_kiwi()

        tab1, tab2, tab3 = st.tabs(["🔑 키워드 분석", "📌 토픽 모델링", "👍 좋아요 분석"])

        with tab1:
            show_keyword_tab(df, kiwi)

        with tab2:
            show_topic_tab(df, kiwi, num_topics)

        with tab3:
            show_likes_tab(df)

        st.divider()
        st.subheader("결과 다운로드")

        # 토픽 결과 병합
        if 'topic_df' in st.session_state:
            topic_result = st.session_state['topic_df'][['댓글내용', '토픽번호', '토픽키워드']]
            download_df = df.merge(topic_result[['댓글내용', '토픽번호', '토픽키워드']],
                                   on='댓글내용', how='left')
        else:
            download_df = df.copy()
            download_df['토픽번호'] = '미분석'
            download_df['토픽키워드'] = '미분석'

        # 좋아요 구간 태그
        def get_tier(likes):
            if likes >= 100: return '핵심여론 (100+)'
            elif likes >= 10: return '공감여론 (10~99)'
            else: return '일반댓글 (~9)'
        download_df['좋아요구간'] = download_df['좋아요수'].apply(get_tier)

        # 컬럼 순서 정리
        base_cols = ['댓글내용', '좋아요수', '작성시간', '작성자', '토픽번호', '토픽키워드', '좋아요구간']
        if '싫어요수' in download_df.columns:
            base_cols.insert(2, '싫어요수')
        download_df = download_df[[c for c in base_cols if c in download_df.columns]]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**총 {len(download_df)}개 댓글** | 컬럼: {', '.join(download_df.columns.tolist())}")
        with col2:
            csv = download_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                "전체 결과 CSV 다운로드",
                csv,
                "analysis_results.csv",
                "text/csv",
                use_container_width=True,
                type="primary"
            )

    else:
        if source == 'youtube':
            st.info("왼쪽 사이드바에 API 키와 YouTube URL을 입력하고 '분석 시작!' 버튼을 눌러주세요.")
        else:
            st.info("왼쪽 사이드바에 네이버 뉴스 URL을 입력하고 '분석 시작!' 버튼을 눌러주세요.")

        st.markdown("""
        ### 분석 항목
        - **🔑 키워드 분석**: 자주 언급되는 단어 + 워드클라우드
        - **📌 토픽 모델링**: 주요 주제 자동 분류 + 대표 댓글
        - **👍 좋아요 분석**: 공감받은 여론 파악 + 구간별 비교
        """)
