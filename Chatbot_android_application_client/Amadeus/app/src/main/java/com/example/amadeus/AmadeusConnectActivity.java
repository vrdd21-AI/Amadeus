package com.example.amadeus;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.util.ArrayList;
import java.util.Locale;

public class AmadeusConnectActivity extends AppCompatActivity {
    private LinearLayout chris;
    private int[] pictures;
    private int id = 0;
    private OutlineTextView outlineTextView;

    private Socket client = null;
    public InputStream inp;
    public OutputStream out;

    private boolean isStop = true;
    private boolean emotionStop = true;
    private boolean emotionStart = false;
    private int state = -1; // -1 : nothing, 0: neutral, 1: negative, 2: positive

    private String IP = "147.46.242.175";
    //private String IP = "192.168.0.16";
    private int PORT = 10004;

    Context cThis;
    Intent SttIntent;
    SpeechRecognizer mRecognizer;
    TextToSpeech tts;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_amadeus_connect);



        outlineTextView = findViewById(R.id.dialog_textview);
        outlineTextView.setText("Okabe Rintaro-san? Nice to meet you.");

        pictures = new int[8];
        pictures[0] = R.drawable.chris_blink_side; // front, 무표정 blink
        pictures[1] = R.drawable.chris_blink_smile; // front, 약한 미소
        pictures[2] = R.drawable.chris_blink_open_smile; // front, 환한 미소

        pictures[3] = R.drawable.chris_normal; // front인데 입은 미소
        pictures[4] = R.drawable.chris_normal_front; // 입 꾹 다뭄
        pictures[5] = R.drawable.chris_normal_side; // 눈이 오른쪽

        pictures[6] = R.drawable.chris_surprise; // front인데 놀란 입
        pictures[7] = R.drawable.chris_wink; // 한쪽 눈 감고 미소

        chris = findViewById(R.id.chris_layout);

        cThis = this;
        SttIntent=new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        SttIntent.putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE,getApplicationContext().getPackageName());
        SttIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US");
        mRecognizer=SpeechRecognizer.createSpeechRecognizer(cThis);
        mRecognizer.setRecognitionListener(listener);


        tts=new TextToSpeech(cThis, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status!=android.speech.tts.TextToSpeech.ERROR){
                    tts.setLanguage(Locale.US);
                }
            }
        });

        emotionChange();
        connect(IP, PORT);
    }

    public void onClickChris(View view){
        System.out.println("음성인식 시작!");
        if(ContextCompat.checkSelfPermission(cThis, Manifest.permission.RECORD_AUDIO)!= PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(AmadeusConnectActivity.this,new String[]{Manifest.permission.RECORD_AUDIO},1);
            //권한을 허용하지 않는 경우
        }else{
            //권한을 허용한 경우
            try {
                mRecognizer.startListening(SttIntent);
            }catch (SecurityException e){e.printStackTrace();}
        }
    }

    private RecognitionListener listener=new RecognitionListener() {
        @Override
        public void onReadyForSpeech(Bundle bundle) {
            //outlineTextView.setText("onReadyForSpeech..........."+"\r\n");
            Log.d("Test", "OnReadyForSpeech");
        }

        @Override
        public void onBeginningOfSpeech() {
            //outlineTextView.setText("지금부터 말을 해주세요...........");
            Log.d("Test", "OnBeginningOfSpeech");
        }

        @Override
        public void onRmsChanged(float v) {

        }

        @Override
        public void onBufferReceived(byte[] bytes) {
            //outlineTextView.setText("onBufferReceived..........."+"\r\n");
        }

        @Override
        public void onEndOfSpeech() {
            //outlineTextView.setText("onEndOfSpeech...........");

            Log.d("Test", "onEndOfSpeech");
        }

        @Override
        public void onError(int i) {
            //outlineTextView.setText("천천히 다시 말해 주세요...........");

            Log.d("Test", "OnError");
        }

        @Override
        public void onResults(Bundle results) {
            String key= "";
            key = SpeechRecognizer.RESULTS_RECOGNITION;
            ArrayList<String> mResult =results.getStringArrayList(key);
            final String[] rs = new String[mResult.size()];
            mResult.toArray(rs);
            //outlineTextView.setText(rs[0]);
            Log.d("Test", rs[0]);
            new Handler().postDelayed(new Runnable() {
                @Override
                public void run() {
                    sendMessage(rs[0]);
                }
            }, 100);
            //FuncVoiceOrderCheck(rs[0]);
            //FuncVoiceOut(rs[0]);
            //mRecognizer.startListening(SttIntent);
            Log.d("Test", "OnResults");
            mRecognizer.stopListening();

        }

        @Override
        public void onPartialResults(Bundle bundle) {
            //outlineTextView.setText("onPartialResults...........");
        }

        @Override
        public void onEvent(int i, Bundle bundle) {
            //outlineTextView.setText("onEvent...........");
        }
    };

    //입력된 음성 메세지 확인 후 동작 처리
    private void FuncVoiceOrderCheck(String VoiceMsg){
        if(VoiceMsg.length()<1)return;

        VoiceMsg=VoiceMsg.replace(" ","");//공백제거
    }

    //음성 메세지 출력용
    private void FuncVoiceOut(String OutMsg){
        if(OutMsg.length()<1)return;

        tts.setPitch(1.0f);//목소리 톤1.0
        tts.setSpeechRate(1.0f);//목소리 속도
        tts.speak(OutMsg,TextToSpeech.QUEUE_FLUSH,null);

        //어플이 종료할때는 완전히 제거

    }

    public void connect(final String ip, final int port){
        new AsyncTask<Void, String, Void>(){
            public String sConnectError = "";
            public String sClientError = "";
            @Override
            protected void onPostExecute(Void aVoid) {

                super.onPostExecute(aVoid);
            }

            @Override
            protected void onProgressUpdate(String... values) {
                super.onProgressUpdate(values);

                if(values == null)
                    return;
                Log.d("Test", values[0]);
                char command = values[0].charAt(0);
                if(command == '0'){

                    state = 0;
                    values[0] = values[0].substring(2, values[0].length());
                    emotionStart = true;
                }
                else if(command == '1'){
                    state = 1;
                    values[0] = values[0].substring(2, values[0].length());
                    emotionStart = true;
                }
                else if(command == '2'){
                    state = 2;
                    values[0] = values[0].substring(2, values[0].length());
                    emotionStart = true;
                }

                outlineTextView.setText(values[0]);
                // 결과값에 따라 어떤 표정을 취해야할지 boolean값 변화
            }

            @Override
            protected Void doInBackground(Void... voids) {
                try{
                    client = new Socket(ip, port);
                    inp = client.getInputStream();
                    out = client.getOutputStream();
                } catch (IOException e){
                    e.printStackTrace();
                    sConnectError = e.getMessage();
                    return null;
                }

                byte[] buffer = new byte[1024];
                int bytes = 0;

                try{
                    isStop = false;

                    while(isStop == false){
                        bytes = inp.read(buffer);
                        if(bytes < 0)
                            break;
                        String sMsg = new String(buffer, 0, bytes);
                        if(TextUtils.isEmpty(sMsg) == false){
                            publishProgress(sMsg); // UI thread
                        }
                    }
                } catch(IOException e){
                    e.printStackTrace();
                    sClientError = e.getMessage();
                }
                return null;
            }
        }.execute();
    }

    public void sendMessage(final String sMsg){
        if(client == null){
            return;
        }

        final String newMsg = "sent " + sMsg;

        new AsyncTask<Void, Integer, Void>(){
            private String sErr = "";
            @Override
            protected Void doInBackground(Void... voids) {
                try{
                    byte[] buffer = newMsg.getBytes();
                    out.write(buffer);
                } catch (IOException e){
                    e.printStackTrace();
                    sErr = e.getMessage();
                }
                return null;
            }

            @Override
            protected void onPostExecute(Void aVoid) {
                super.onPostExecute(aVoid);
            }
        }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
        // executeOnExecutor는 무슨 의미지
    }

    public void emotionChange(){
        new AsyncTask<Void, Void, Void>(){
            @Override
            protected void onProgressUpdate(Void... values) {
                super.onProgressUpdate(values);
                try {
                    switch (state) {
                        case 0:
                            changeImage(3);
                            //Thread.sleep(300);
                            break;
                        case 1:
                            changeImage(4);
                            break;
                        case 2:
                            changeImage(7);
                            break;
                        default:
                            break;
                    }
                } catch (Exception e){
                    e.printStackTrace();
                }
            }

            @Override
            protected Void doInBackground(Void... voids) {

                emotionStop = false;
                while(emotionStop == false){
                    if(emotionStart){
                        emotionStart = false;
                        Log.d("Test", "here");
                        onProgressUpdate();
                    }
                }
                return null;
            }
        }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    }

    //카톡으로 이동을 했는데 음성인식 어플이 종료되지 않아 계속 실행되는 경우를 막기위해 어플 종료 함수
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if(tts!=null){
            tts.stop();
            tts.shutdown();
            tts=null;
        }
        if(mRecognizer!=null){
            mRecognizer.destroy();
            mRecognizer.cancel();
            mRecognizer=null;
        }

        try{
            if(client != null) {
                client.close();
                Log.d("Test", "closed");
            }
        } catch(IOException e){
            e.printStackTrace();
        }

    }

    private void changeImage(final int resId){
        chris.setBackgroundResource(pictures[resId]);
    }
}
