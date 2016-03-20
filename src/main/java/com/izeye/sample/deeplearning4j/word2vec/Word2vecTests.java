package com.izeye.sample.deeplearning4j.word2vec;

import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Collection;

/**
 * Created by izeye on 16. 3. 20..
 */
public class Word2vecTests {

	public static void main(String[] args) {
		try {
			File file = new ClassPathResource("raw_sentences.txt").getFile();
			SentenceIterator sentenceIterator = new BasicLineIterator(file);
			DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
			tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

			Word2Vec word2Vec = new Word2Vec.Builder()
					.minWordFrequency(5)
					.iterations(1)
					.layerSize(100)
					.seed(42)
					.windowSize(5)
					.iterate(sentenceIterator)
					.tokenizerFactory(tokenizerFactory).build();
			word2Vec.fit();

			Collection<String> words = word2Vec.wordsNearest("day", 10);
			System.out.println(words);

			// TODO: How to use this?
//			UiServer uiServer = UiServer.getInstance();
//			System.out.println("UI server is running or port " + uiServer.getPort());
		} catch (FileNotFoundException ex) {
			throw new RuntimeException(ex);
		} catch (Exception ex) {
			throw new RuntimeException(ex);
		}
	}
	
}
