#!/bin/bash
# Copyright 2020  Delft University of Technology (Siyuan Feng)
# Copyright 2020  Johns Hopkins University (Author: Piotr Żelasko)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eou pipefail

stage=1 # stage 0 runs only once and for all crosslingual experiments
stop_stage=500
gen_ali=true
train_tri1_stage=-10
train_tri3_stage=-10
train_tri4_stage=-10
train_tri5_stage=-10
extract_feat_nj=8
early_train_nj=60
train_nj=100
phone_ngram_order=2
word_ngram_order=3
# When phone_tokens is false, we will use regular phones (e.g. /ae/) as our basic phonetic unit.
# Otherwise, we will split them up to characters (e.g. /ae/ -> /a/, /e/).
phone_tokens=false
# When use_word_supervisions is true, we will add a language suffix to each word
# (e.g. "cat" -> "cat_English") and use these transcripts to train a word-level
# language model and the lang directory for model training.
# Otherwise, we will use phones themselves as "fake words"
# (e.g. text will be "k ae t" instead of "cat_English")
use_word_supervisions=false

# Acoustic model parameters
numLeavesTri1=1000
numGaussTri1=10000
numLeavesTri2=1000
numGaussTri2=20000
numLeavesTri3=6000
numGaussTri3=75000
numLeavesMLLT=6000
numGaussMLLT=75000
numLeavesSAT=6000
numGaussSAT=75000

lang_to_recog=Czech # Czech, ..., Thai, 101, 103, ... 404

. cmd.sh
. utils/parse_options.sh
. path.sh


langs_config=conf/experiments/crossling_eval_${lang_to_recog}.conf
if [ $langs_config ]; then
  # shellcheck disable=SC1090
  source $langs_config
  echo "Getting language config from $langs_config"
else
  # BABEL TRAIN:
  # Amharic - 307
  # Bengali - 103
  # Cantonese - 101
  # Javanese - 402
  # Vietnamese - 107
  # Zulu - 206
  # BABEL TEST:
  # Georgian - 404
  # Lao - 203
  babel_langs="307 103 101 402 107 206 404 203"
  babel_recog="${babel_langs}"
  gp_langs="Czech French Mandarin Spanish Thai"
  gp_recog="${gp_langs}"
  gp_path="/export/corpora5/GlobalPhone"
  mboshi_train=false
  mboshi_recog=false
  gp_romanized=false
fi
###Globalphone####
#Czech       S0196
#French      S0197
#Spanish     S0203
#Mandarin    S0193
#Thai        S0321


local/install_shorten.sh
echo "$0: langs_config:$langs_config"
echo "$0: babel_langs=$babel_langs, babel_recog=$babel_recog, gp_langs=$gp_langs, gp_recog=$gp_recog"

train_set=""
dev_set=""
for l in ${babel_langs}; do
  train_set="$l/data/train_${l} ${train_set}"
  dev_set="$l/data/dev_${l} ${dev_set}"
done
train_set_data=""
dev_set_data=""
for l in ${babel_langs}; do
  train_set_data="data/$l/data/train_${l} ${train_set_data}"
  dev_set_data="data/$l/data/dev_${l} ${dev_set_data}"
done
for l in ${gp_langs}; do
  train_set="GlobalPhone/gp_${l}_train ${train_set}"
  dev_set="GlobalPhone/gp_${l}_dev ${dev_set}"
done
for l in ${gp_langs}; do
  train_set_data="data/GlobalPhone/gp_${l}_train ${train_set_data}"
  dev_set_data="data/GlobalPhone/gp_${l}_dev ${dev_set_data}"
done
train_set=${train_set%% }
dev_set=${dev_set%% }
train_set_data=${train_set_data%% }
dev_set_data=${dev_set_data%% }

recog_set=""
for l in ${babel_recog} ${gp_recog}; do
  recog_set="eval_${l} ${recog_set}"
done
recog_set=${recog_set%% }

echo "Training data directories: ${train_set[*]}"
echo "Dev data directories: ${dev_set[*]}"
echo "Eval data directories: ${recog_set[*]}"

full_train_set=train
full_dev_set=dev

function langname() {
  # Utility
  echo "$(basename "$1")"
}

phone_token_opt='--phones'
if [ $phone_tokens = true ]; then
  phone_token_opt='--phone-tokens'
fi

dir_suffix=_crosslang_recog_${lang_to_recog} # denotes which language set as evaluation, so the remaining 12 languages are for training
exp_dir_root=exp/gmm${dir_suffix}
# This step will create the data directories for GlobalPhone and Babel languages.
# It's also going to use LanguageNet G2P models to convert text into phonetic transcripts.
# Depending on the settings, it will either transcribe into phones, e.g. ([m], [i:], [t]), or
# phonetic tokens, e.g. (/m/, /i/, /:/, /t/).
# The Kaldi "text" file will consist of these phonetic sequences, as we're trying to build
# a universal IPA recognizer.
# The lexicons are created separately for each split as an artifact from the ESPnet setup.
if (($stage <= 0)) && (($stop_stage > 0)); then
  # we still go through 13 langs even if we know we won't merge eval languange into merged training data at data/univseral_crosslang_recog_$eval. 
  #After this stage is done running once, omit it when running new crosslingual AM training.
  # that's why we use babel_/gp_langs_whole below
  echo "stage 0: Setting up individual languages"
  babel_langs_whole="101 103 107 203 206 307 402 404"
  babel_recog_whole="101 103 107 203 206 307 402 404" #"$babel_langs_whole"
  gp_langs_whole="Czech French Spanish Mandarin Thai"
  gp_recog_whole="Czech French Spanish Mandarin Thai" #"$gp_langs_whole"
  echo "babel_langs_whole: $babel_langs_whole"
  echo "gp_langs_whole: $gp_langs_whole"
  local/setup_languages.sh \
    --langs "${babel_langs_whole}" \
    --recog "${babel_recog_whole}" \
    --gp-langs "${gp_langs_whole}" \
    --gp-recog "${gp_recog_whole}" \
    --mboshi-train "${mboshi_train}" \
    --mboshi-recog "${mboshi_recog}" \
    --gp-romanized "${gp_romanized}" \
    --gp-path "${gp_path}" \
    --phone_token_opt "${phone_token_opt}" \
    --multilang true
  train_set_whole=""
  dev_set_whole=""
  for l in ${babel_langs_whole}; do
    train_set_whole="$l/data/train_${l} ${train_set_whole}"
    dev_set_whole="$l/data/dev_${l} ${dev_set_whole}"
  done
  for l in ${gp_langs_whole}; do
    train_set_whole="GlobalPhone/gp_${l}_train ${train_set_whole}"
    dev_set_whole="GlobalPhone/gp_${l}_dev ${dev_set_whole}"
  done
  train_set_whole=${train_set_whole%% }
  dev_set_whole=${dev_set_whole%% }
  recog_set_whole=""
  for l in ${babel_recog_whole} ${gp_recog_whole}; do
    recog_set_whole="eval_${l} ${recog_set_whole}"
  done
  recog_set_whole=${recog_set_whole%% }
  echo "$0: train_set_whole:$train_set_whole, dev_set_whole:$dev_set_whole, recog_set_whole:$recog_set_whole"
  for x in ${train_set_whole} ${dev_set_whole} ${recog_set_whole}; do
    sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp
  done
fi

# Repair step if you changed your mind regarding word supervisions after running a few steps...

if $use_word_supervisions; then
  for data_dir in ${train_set}; do
    if [ -f data/$data_dir/text.bkp_suffix ]; then
      # replace IPA text with normal text (word having language suffix e.g. _Czech
      #cp data/$data_dir/text.bkp data/$data_dir/text
      cp data/$data_dir/text.bkp_suffix data/$data_dir/text
    fi
  done
else
  for data_dir in ${train_set}; do
    if [ -f data/$data_dir/text.bkp_suffix ]; then
      # replace IPA text with normal text (word having language suffix e.g. _Czech
      #cp data/$data_dir/text.bkp data/$data_dir/text
      cp data/$data_dir/text.ipa data/$data_dir/text
    fi
  done
fi

# Here we will combine the lexicons for train/dev/test splits into a single lexicon for each language.
if ((stage <= 1)) && ((stop_stage > 1)); then
  for data_dir in ${train_set}; do
    lang_name=$(langname $data_dir)
    mkdir -p data/local${dir_suffix}/$lang_name
    python3 local/combine_lexicons.py \
      data/$data_dir/lexicon_ipa.txt \
      data/${data_dir//train/dev}/lexicon_ipa.txt \
      data/${data_dir//train/eval}/lexicon_ipa.txt \
      >data/$data_dir/lexicon_ipa_all.txt
    python3 local/prepare_lexicon_dir.py $phone_token_opt data/$data_dir/lexicon_ipa_all.txt data/local${dir_suffix}/$lang_name
  done
fi

# We use the per-language lexicons to find the set of phones/phonetic tokens in every language and combine
# them again to obtain a multilingual "dummy" lexicon of the form:
# a a
# b b
# c c
# ...
# When that is ready, we train a multilingual phone-level language model (i.e. phonotactic model),
# that will be used to compile the decoding graph and to score each ASR system.
if ((stage <= 2)) && ((stop_stage > 2)); then
  # in crosslingual case, LM output dir is e.g. data/ipa_lm_crosslang_recog_Czech/train_all showing we're not using Czech text in training LM 
  local/prepare_ipa_lm.sh \
    --output-dir-suffix "${dir_suffix}" \
    --train-set "$train_set" \
    --phone_token_opt "$phone_token_opt" \
    --order "$phone_ngram_order"
  lexicon_list=$(find data/ipa_lm${dir_suffix}/train -name lexiconp.txt)
  mkdir -p data/local${dir_suffix}/dict_combined/local
  python3 local/combine_lexicons.py $lexicon_list >data/local${dir_suffix}/dict_combined/local/lexiconp.txt
  python3 local/prepare_lexicon_dir.py data/local${dir_suffix}/dict_combined/local/lexiconp.txt data/local${dir_suffix}/dict_combined
  utils/prepare_lang.sh \
    --position-dependent-phones false \
    data/local${dir_suffix}/dict_combined "<unk>" data/local${dir_suffix}/dict_combined data/lang_combined${dir_suffix}
  PHONE_LM=data/ipa_lm${dir_suffix}/train_all/srilm.o${phone_ngram_order}g.kn.gz

  if [ "$phone_ngram_order" = "2" ];then
    lm_order_suffix=""
  else
    lm_order_suffix="_${phone_ngram_order}gram"
  fi
  utils/format_lm.sh data/lang_combined${dir_suffix} "$PHONE_LM" data/local${dir_suffix}/dict_combined/lexicon.txt data/lang_combined${dir_suffix}_test${lm_order_suffix}
fi

if (($stage <= 3)) && (($stop_stage > 3)); then
  #  We will generate a universal lexicon dir: data/local${dir_suffix}/lang_universal and
  #                      a universal lang dir: data/lang_universal${dir_suffix};
  #  data/lang_universal${dir_suffix}/words.txt come from multiple languages and each with a language suffix like _101.
  #  Pronunciations in data/lang_universal${dir_suffix}/phones/align_lexicon.txt use IPA phone symbols, same as in monolingual recipe
  mkdir -p data/local${dir_suffix}/lang_universal
  for data_dir in ${train_set}; do
    dev_data_dir=${data_dir//train/dev}
    eval_data_dir=${data_dir//train/eval}
    lang_name="$(langname $data_dir)"
    data_contain_lexicon_ipa_suffix=../v1_multilang/data/
    python3 local/combine_lexicons.py \
      $data_contain_lexicon_ipa_suffix/$data_dir/lexicon_ipa_suffix.txt \
      $data_contain_lexicon_ipa_suffix/$dev_data_dir/lexicon_ipa_suffix.txt \
      $data_contain_lexicon_ipa_suffix/$eval_data_dir/lexicon_ipa_suffix.txt \
      >data/local${dir_suffix}/lang_universal/lexicon_ipa_suffix_${lang_name}.txt
#    cp data/$data_dir/lexicon_ipa_suffix.txt data/local${dir_suffix}/lang_universal/lexicon_ipa_suffix_${lang_name}.txt
  done
  # Create a language-universal lexicon; each word has a language-suffix like "word_English word_Czech";
  # Because of that we can just concatenate and sort the lexicons.
  cat data/local${dir_suffix}/lang_universal/lexicon_ipa_suffix*.txt |
    sort \
      >data/local${dir_suffix}/lang_universal/lexicon_ipa_suffix_universal.txt
  # Create a regular Kaldi dict dir using the combined lexicon.
  python3 local/prepare_lexicon_dir.py \
    $phone_token_opt \
    data/local${dir_suffix}/lang_universal/lexicon_ipa_suffix_universal.txt \
    data/local${dir_suffix}/lang_universal
  # Create a regular Kaldi lang dir using the combined lexicon.
  utils/prepare_lang.sh \
    --position-dependent-phones false \
    --share-silence-phones true \
    data/local${dir_suffix}/lang_universal '<unk>' data/local${dir_suffix}/tmp.lang_universal data/lang_universal${dir_suffix}
  # Train the LM and evaluate on the dev set transcripts
  local/prepare_word_lm.sh \
    --train-set "$train_set" \
    --order "$word_ngram_order" \
    --output-dir-suffix "${dir_suffix}"
  WORD_LM=data/word_lm${dir_suffix}/train_all/srilm.o${word_ngram_order}g.kn.gz
  utils/format_lm.sh data/lang_universal${dir_suffix} "$WORD_LM" data/local${dir_suffix}/lang_universal/lexicon_ipa_suffix_universal.txt data/lang_universal${dir_suffix}_test
fi

if (($stage <= 4)) && (($stop_stage > 4)); then
  # Feature extraction
  for data_dir in ${train_set}; do
    (
    # If a certain language's mfcc has been extracted in previous crosslingual experiments, do not extract again
    if [ ! -f data/$data_dir/cmvn.scp ]; then
      lang_name=$(langname $data_dir)
      steps/make_mfcc.sh \
        --cmd "$train_cmd" \
        --nj $extract_feat_nj \
        --write_utt2num_frames true \
        "data/$data_dir" \
        "exp/make_mfcc/$data_dir" \
        mfcc
      utils/fix_data_dir.sh data/$data_dir
      steps/compute_cmvn_stats.sh data/$data_dir exp/make_mfcc/$lang_name mfcc/$lang_name
    fi
    ) &
    sleep 2
  done
  wait
fi

if (($stage <= 5)) && (($stop_stage > 5)); then
  echo "combine data dirs to a universal data dir in data/universal${dir_suffix}"
  echo "train_set_data: $train_set_data"
  utils/combine_data.sh data/universal${dir_suffix}/train $train_set_data
  utils/validate_data_dir.sh data/universal${dir_suffix}/train || exit 1
  echo "$train_set" >data/universal${dir_suffix}/train/original_data_dirs.txt
fi

if (($stage <= 6)) && (($stop_stage > 6)); then
  # Prepare data dir subsets for monolingual training
  numutt=$(cat data/universal${dir_suffix}/train/feats.scp | wc -l)
  if [ $numutt -gt 50000 ]; then
    utils/subset_data_dir.sh data/universal${dir_suffix}/train 50000 data/subsets/50k/universal${dir_suffix}/train
  else
    mkdir -p "$(dirname data/subsets/50k/universal${dir_suffix}/train)"
    ln -s "$(pwd)/data/universal${dir_suffix}/train" "data/subsets/50k/universal${dir_suffix}/train"
  fi
  if [ $numutt -gt 100000 ]; then
    utils/subset_data_dir.sh data/universal${dir_suffix}/train 100000 data/subsets/100k/universal${dir_suffix}/train
  else
    mkdir -p "$(dirname data/subsets/100k/universal${dir_suffix}/train)"
    ln -s "$(pwd)/data/universal${dir_suffix}/train" "data/subsets/100k/universal${dir_suffix}/train"
  fi
  if [ $numutt -gt 200000 ]; then
    utils/subset_data_dir.sh data/universal${dir_suffix}/train 200000 data/subsets/200k/universal${dir_suffix}/train
  else
    mkdir -p "$(dirname data/subsets/200k/universal${dir_suffix}/train)"
    ln -s "$(pwd)/data/universal${dir_suffix}/train" "data/subsets/200k/universal${dir_suffix}/train"
  fi
fi

lang=data/lang_combined${dir_suffix}_test
if $use_word_supervisions; then
  lang=data/lang_universal_${dir_suffix}_test
fi

data_dir=universal${dir_suffix}/train
if (($stage <= 7)) && (($stop_stage > 7)); then
  # Mono training
  expdir=$exp_dir_root/mono
  steps/train_mono.sh \
    --nj $early_train_nj --cmd "$train_cmd" \
    data/subsets/50k/$data_dir \
    $lang $expdir
fi

if (($stage <= 8)) && (($stop_stage > 8)); then
  # Tri1 training
  steps/align_si.sh \
    --nj $early_train_nj --cmd "$train_cmd" \
    data/subsets/100k/$data_dir \
    $lang \
    $exp_dir_root/mono \
    $exp_dir_root/mono_ali_100k

  steps/train_deltas.sh \
    --stage $train_tri1_stage \
    --cmd "$train_cmd" \
    $numLeavesTri1 \
    $numGaussTri1 \
    data/subsets/100k/$data_dir \
    $lang \
    $exp_dir_root/mono_ali_100k \
    $exp_dir_root/tri1
fi

if (($stage <= 9)) && (($stop_stage > 9)); then
  # Tri2 training
  steps/align_si.sh \
    --nj $early_train_nj --cmd "$train_cmd" \
    data/subsets/200k/$data_dir \
    $lang \
    $exp_dir_root/tri1 \
    $exp_dir_root/tri1_ali_200k

  steps/train_deltas.sh \
    --cmd "$train_cmd" $numLeavesTri2 $numGaussTri2 \
    data/subsets/200k/$data_dir \
    $lang \
    $exp_dir_root/tri1_ali_200k \
    $exp_dir_root/tri2
fi

if (($stage <= 10)) && (($stop_stage > 10)); then
  # Tri3 training
  steps/align_si.sh \
    --nj $train_nj --cmd "$train_cmd" \
    data/$data_dir \
    $lang \
    $exp_dir_root/tri2 \
    $exp_dir_root/tri2_ali

  steps/train_deltas.sh \
    --stage $train_tri3_stage \
    --cmd "$train_cmd" $numLeavesTri3 $numGaussTri3 \
    data/$data_dir \
    $lang \
    $exp_dir_root/tri2_ali \
    $exp_dir_root/tri3
fi

if (($stage <= 11)) && (($stop_stage > 11)); then
  # Tri4 training
  if $gen_ali; then
  steps/align_si.sh \
    --nj $train_nj --cmd "$train_cmd" \
    data/$data_dir \
    $lang \
    $exp_dir_root/tri3 \
    $exp_dir_root/tri3_ali
  fi
  steps/train_lda_mllt.sh \
    --stage $train_tri4_stage \
    --cmd "$train_cmd" \
    $numLeavesMLLT \
    $numGaussMLLT \
    data/$data_dir \
    $lang \
    $exp_dir_root/tri3_ali \
    $exp_dir_root/tri4
fi

if (($stage <= 12)) && (($stop_stage > 12)); then
  # Tri5 training
  if $gen_ali; then
  steps/align_si.sh \
    --nj $train_nj --cmd "$train_cmd" \
    data/$data_dir \
    $lang \
    $exp_dir_root/tri4 \
    $exp_dir_root/tri4_ali
  fi
  steps/train_sat.sh \
    --stage $train_tri5_stage \
    --cmd "$train_cmd" \
    $numLeavesSAT \
    $numGaussSAT \
    data/$data_dir \
    $lang \
    $exp_dir_root/tri4_ali \
    $exp_dir_root/tri5
fi

if (($stage <= 13)) && (($stop_stage > 13)); then
  # Tri5 alignments
  steps/align_fmllr.sh \
    --nj $train_nj --cmd "$train_cmd" \
    data/$data_dir \
    $lang \
    $exp_dir_root/tri5 \
    $exp_dir_root/tri5_ali
fi

# Uncomment this if you intend to train Chain TDNNF AM in next steps
#bash local/chain_crosslang/tuning/run_tdnn_1g.sh
