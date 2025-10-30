"""Task 7a: TagCrowd Word Cloud - Telecom Call Records"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv('telecom_customer_call_records_100.csv')
print(f"Dataset: {df.shape[0]} records, {df['Place'].nunique()} cities")

# Combine text with weights
def cat_dur(d): return ["Short_Call","Medium_Call","Long_Call","Very_Long_Call"][min(d//500, 3) if d < 2500 else 3]
text_data = df['Place'].tolist()*3 + df['Tower_ID'].astype(str).tolist() + df['Customer_ID'].astype(str).tolist() + df['Call_Duration_sec'].apply(cat_dur).tolist()*2

# Generate and save word cloud
wc = WordCloud(width=1600, height=900, background_color='white', colormap='viridis', max_words=200, relative_scaling=0.5, random_state=42, collocations=False).generate(' '.join(text_data))
plt.figure(figsize=(16,9), dpi=100)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('TagCrowd-Style Word Cloud: Telecom Customer Call Records Analysis', fontsize=20, fontweight='bold', pad=20)
plt.text(0.5, -0.05, f'Dataset: {len(df)} Records | Cities: {df["Place"].nunique()}', ha='center', va='top', transform=plt.gca().transAxes, fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.savefig('7a_tagcrowd_wordcloud_output.png', dpi=300, bbox_inches='tight')
plt.close()

# Statistics
print("\nTOP 15 WORDS:", *[f"{w}: {f:.3f}" for w,f in sorted(wc.words_.items(), key=lambda x: x[1], reverse=True)[:15]], sep="\n  ")
print("\nCITY DISTRIBUTION:", *[f"{c}: {n} ({n/len(df)*100:.1f}%)" for c,n in df['Place'].value_counts().items()], sep="\n  ")
print(f"\nCALL STATS: Avg={df['Call_Duration_sec'].mean():.0f}s, Med={df['Call_Duration_sec'].median():.0f}s, Range={df['Call_Duration_sec'].min()}-{df['Call_Duration_sec'].max()}s")
print("[SUCCESS] Task 7a completed!")
